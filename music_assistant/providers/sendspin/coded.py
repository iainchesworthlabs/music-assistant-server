"""
Coded (undecoded) E-AC-3 playback for Sendspin players that list ``_iclforge_player@v1``.

Design X (see the fork's design notes): the playback session keeps decoding the item to PCM
for the group's timeline and for members that cannot take coded audio, and this tap sends the
same item's E-AC-3 access units, undecoded, to the members that can. Burst ``k`` is stamped
with the timeline start of the item plus ``k * 1536`` samples, the PCM timeline of the same item.

Each group member gets one of three variants, chosen from what its board states:

- FULL: the demuxed stream, objects intact.
- STEREO: a two-channel E-AC-3 re-encode (the board lists fewer coded channels than the stream has).
- PCM: the ordinary ``player@v1`` path, for everything else.

Known limits of this first slice: the item is a single-item session (no queue advance or crossfade
inside it), seeking is frame accurate rather than sample accurate for the coded side, and members
that join mid-item get PCM only.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

from music_assistant_models.enums import ContentType

from music_assistant.helpers.passthrough import (
    PassthroughStreamError,
    get_eac3_demux_stream,
    is_coded_passthrough_candidate,
    iter_eac3_access_units,
)

from .ma_role import eac3
from .ma_role import role as iclforge_role

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Iterable

    from aiosendspin.server.client import SendspinClient
    from music_assistant_models.streamdetails import StreamDetails

DATA_TYPE = "eac3"
# the sink holds a coded start until its clock converged (about 2 s after activation)
CODED_START_HOLD_S = 2.5
STATE_TIMEOUT_S = 10.0
STEREO_BIT_RATE = 192_000
# bursts demuxed ahead of the timeline, per variant (32 ms each)
_BURST_QUEUE_SIZE = 96

# the one place the role is registered: importing this module is enough
iclforge_role.register()


class CodedVariant(StrEnum):
    """What one group member is sent for a coded item."""

    FULL = "full"
    STEREO = "stereo"
    PCM = "pcm"


def choose_member_variant(
    support: iclforge_role.IclforgePlayerSupport | None, channels: int, sample_rate: int
) -> CodedVariant:
    """
    Choose the variant for a member from the support its board stated.

    A limit of 0, or none, is unstated and means no limit. ``max_objects`` does not change
    the choice: a board with fewer objects than the stream still gets the full stream and
    reproduces what it can.

    :param support: The ``_iclforge_player@v1_support`` object, None if the board has no such role.
    :param channels: Channel count of the source stream.
    :param sample_rate: Sample rate of the source stream.
    """
    if support is None or DATA_TYPE not in support.data_types:
        return CodedVariant.PCM
    if support.sample_rates and sample_rate not in support.sample_rates:
        return CodedVariant.PCM
    limit = support.max_coded_channels.get(DATA_TYPE, 0)
    if limit and limit < channels:
        return CodedVariant.STEREO if limit >= 2 else CodedVariant.PCM
    return CodedVariant.FULL


def coded_role(client: SendspinClient) -> iclforge_role.IclforgePlayerRole | None:
    """Return the client's active ``_iclforge_player@v1`` role, None while it is not active."""
    role = client.role(iclforge_role.ROLE_ID)
    return role if isinstance(role, iclforge_role.IclforgePlayerRole) else None


def client_support(client: SendspinClient) -> iclforge_role.IclforgePlayerSupport | None:
    """Return the role support a client stated in its hello, if any."""
    return getattr(client.info, f"{iclforge_role.FAMILY}_support", None)


@dataclass(slots=True)
class _Lane:
    """One variant's burst feed and the members it goes to."""

    variant: CodedVariant
    clients: list[SendspinClient]
    queue: asyncio.Queue[eac3.Burst | None] = field(
        default_factory=lambda: asyncio.Queue(maxsize=_BURST_QUEUE_SIZE)
    )
    task: asyncio.Task[None] | None = None
    sent: int = 0
    sent_bytes: int = 0
    next_samples: int = 0
    held: eac3.Burst | None = None


class CodedTap:
    """Sends one queue item's coded bursts to the members that take them."""

    def __init__(
        self,
        logger: logging.Logger,
        path: str,
        seek_position: float,
        channels: int,
        sample_rate: int,
        lanes: list[_Lane],
        pcm_clients: list[SendspinClient],
    ) -> None:
        """Initialize the tap, use :func:`build_coded_tap` instead."""
        self._logger = logger
        self._path = path
        self._seek = seek_position
        self._channels = channels
        self._sample_rate = sample_rate
        self._lanes = lanes
        self.pcm_clients = pcm_clients
        self._timeline_start_us: int | None = None
        self._produced_us = 0
        self.first_lead_us: int | None = None

    @property
    def variants(self) -> dict[str, str]:
        """Return the variant per member id, for logs and the player's state."""
        result = {c.client_id: lane.variant.value for lane in self._lanes for c in lane.clients}
        result.update({c.client_id: CodedVariant.PCM.value for c in self.pcm_clients})
        return result

    async def prepare(self) -> None:
        """
        Wait for the coded roles, start their streams and the demux tasks.

        Call after the modes are set and before the first commit: the sinks hold a coded start
        until their clock has converged, so the first commit is held back by ``CODED_START_HOLD_S``.

        :raises TimeoutError: If a member's role never reports its state.
        """
        roles = [
            (lane, client, coded_role(client)) for lane in self._lanes for client in lane.clients
        ]
        for _, client, role in roles:
            if role is None:
                raise iclforge_role.RoleError(f"{client.client_id}: coded role not active")
            await role.wait_for_state(STATE_TIMEOUT_S)
        await asyncio.sleep(CODED_START_HOLD_S)
        for _, _, role in roles:
            assert role is not None  # for type checking
            role.start_stream(DATA_TYPE, self._sample_rate)
        for lane in self._lanes:
            lane.task = asyncio.create_task(self._demux(lane))
            lane.task.add_done_callback(self._log_task_end)

    async def on_commit(self, commit_start_us: int, duration_us: int, now_us: int) -> None:
        """
        Send the bursts that start inside the PCM span just committed.

        :param commit_start_us: Timeline timestamp the committed PCM starts at.
        :param duration_us: Duration of the committed PCM.
        :param now_us: The server clock now, to report the lead of the first burst.
        """
        if self._timeline_start_us is None:
            self._timeline_start_us = commit_start_us
            self.first_lead_us = commit_start_us - now_us
            self._logger.debug(
                "Coded tap starts at %d us, %.0f ms from now",
                commit_start_us,
                self.first_lead_us / 1000,
            )
        self._produced_us += duration_us
        for lane in self._lanes:
            await self._send_due(lane)

    async def close(self, *, drain: bool = False) -> None:
        """
        End the coded streams and stop the demux tasks.

        :param drain: Send what is still queued first (end of the item), else drop it.
        """
        for lane in self._lanes:
            if lane.task is not None:
                lane.task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await lane.task
            if drain:
                self._produced_us = 1 << 62
                with contextlib.suppress(Exception):
                    await self._send_due(lane)
            self._logger.info(
                "Coded %s lane: %d bursts, %d bytes, first burst %s ms ahead, to %s",
                lane.variant.value,
                lane.sent,
                lane.sent_bytes,
                None if self.first_lead_us is None else self.first_lead_us // 1000,
                [c.client_id for c in lane.clients],
            )
            for client in lane.clients:
                if (role := coded_role(client)) is not None:
                    with contextlib.suppress(Exception):
                        role.end_stream()

    async def _send_due(self, lane: _Lane) -> None:
        """Send queued bursts of a lane whose start lies inside the committed timeline."""
        assert self._timeline_start_us is not None
        while True:
            burst = lane.held
            lane.held = None
            if burst is None:
                try:
                    burst = lane.queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
            if burst is None:
                return
            offset_us = lane.next_samples * 1_000_000 // self._sample_rate
            if offset_us >= self._produced_us:
                lane.held = burst
                return
            stamp = self._timeline_start_us + offset_us
            for client in lane.clients:
                role = coded_role(client)
                if role is not None and role.stream_started:
                    role.send_burst(burst, stamp)
            lane.next_samples += burst.samples
            lane.sent += 1
            lane.sent_bytes += len(burst.payload)

    async def _demux(self, lane: _Lane) -> None:
        """Feed a lane's queue with bursts built from the item's access units."""
        builder = eac3.Eac3BurstBuilder()
        stereo = lane.variant == CodedVariant.STEREO
        stream = get_eac3_demux_stream(
            self._path, self._seek, stereo_bit_rate=STEREO_BIT_RATE if stereo else None
        )
        try:
            async for unit in iter_eac3_access_units(_chunks_of(stream)):
                if (burst := builder.push(unit)) is not None:
                    await lane.queue.put(burst)
        except (PassthroughStreamError, eac3.Eac3Error) as err:
            self._logger.error("Coded %s stream of %s broke: %s", lane.variant, self._path, err)
        finally:
            await lane.queue.put(None)
            await stream.aclose()

    def _log_task_end(self, task: asyncio.Task[None]) -> None:
        if not task.cancelled() and (err := task.exception()):
            self._logger.error("Coded demux task failed: %s", err)


async def _chunks_of(stream: AsyncGenerator[bytes]) -> AsyncGenerator[bytes]:
    async for chunk in stream:
        yield chunk


async def build_coded_tap(
    logger: logging.Logger,
    clients: Iterable[SendspinClient],
    streamdetails: StreamDetails | None,
    seek_position: float = 0,
) -> CodedTap | None:
    """
    Set every member's mode for the item and return the tap, or None for a PCM-only item.

    Members that list ``_iclforge_player@v1`` are put in coded mode when they get a coded
    variant, and back in PCM mode otherwise (including for every non coded item), so a previous
    coded item never leaves a board without its ``player@v1``. Must run before the group's
    first commit.

    :param logger: Logger of the session.
    :param clients: The clients of the playing group.
    :param streamdetails: Stream details of the item the session starts with.
    :param seek_position: Start position of the item in seconds.
    """
    clients = list(clients)
    candidate = streamdetails is not None and is_coded_passthrough_candidate(streamdetails)
    audio_format = streamdetails.audio_format if streamdetails else None
    channels = audio_format.channels if audio_format else 2
    sample_rate = audio_format.sample_rate if audio_format else 48000
    if audio_format is not None and ContentType.EAC3 not in {
        audio_format.content_type,
        audio_format.codec_type,
    }:
        candidate = False
    by_variant: dict[CodedVariant, list[SendspinClient]] = {v: [] for v in CodedVariant}
    for client in clients:
        support = client_support(client)
        if support is None:
            by_variant[CodedVariant.PCM].append(client)
            continue
        variant = (
            choose_member_variant(support, channels, sample_rate) if candidate else CodedVariant.PCM
        )
        by_variant[variant].append(client)
        try:
            await iclforge_role.set_mode(client, "pcm" if variant == CodedVariant.PCM else "coded")
        except (iclforge_role.RoleError, TimeoutError) as err:
            logger.warning("Could not set the playback mode of %s: %s", client.client_id, err)
            if variant != CodedVariant.PCM:
                by_variant[variant].remove(client)
                by_variant[CodedVariant.PCM].append(client)
    lanes = [
        _Lane(variant=variant, clients=members)
        for variant in (CodedVariant.FULL, CodedVariant.STEREO)
        if (members := by_variant[variant])
    ]
    if not lanes or streamdetails is None or not isinstance(streamdetails.path, str):
        return None
    return CodedTap(
        logger,
        streamdetails.path,
        seek_position,
        channels,
        sample_rate,
        lanes,
        by_variant[CodedVariant.PCM],
    )
