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
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from aiosendspin.models import AudioCodec
from aiosendspin.models.core import StreamClearMessage, StreamClearPayload
from aiosendspin.server.audio import AudioFormat as SendspinAudioFormat
from music_assistant_models.enums import ContentType

from music_assistant.helpers.passthrough import (
    PassthroughStreamError,
    get_eac3_demux_stream,
    is_coded_passthrough_candidate,
    iter_eac3_access_units,
    probe_eac3_stream,
)

from .ma_role import eac3
from .ma_role import role as iclforge_role

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Coroutine, Iterable

    from aiosendspin.models.player import SupportedAudioFormat
    from aiosendspin.server.client import SendspinClient
    from music_assistant_models.streamdetails import StreamDetails

DATA_TYPE = "eac3"
PLAYER_ROLE_ID = iclforge_role.PLAYER_ROLE_ID
PLAYER_FAMILY = "player"
# the sink holds a coded start until its clock converged (about 2 s after activation)
CODED_START_HOLD_S = 2.5
# bursts every lane must have ready before the first commit, and how long to wait for them
PRIME_BURSTS = 24
PRIME_TIMEOUT_S = 3.0
# nothing in the teardown of a tap may wait longer than this
CLOSE_TIMEOUT_S = 1.5
# a coded item resumed this close to its end starts over from the beginning
RESTART_THRESHOLD_S = 2.0
STATE_TIMEOUT_S = 3.0
# a member that is still connected may be busy ending its previous stream: it gets this much
# longer for its role state before it is given up on
STATE_PATIENCE_S = 12.0
# a board that decodes 15-object material at 192 kb/s stereo at its limit (the C6) gets less to chew
STEREO_BIT_RATE = 128_000
# the push stream cuts PCM into 25 ms chunks; a board states the largest message it takes
PCM_CHUNK_US = 25_000
CHUNK_OVERHEAD_BYTES = 40
# added to the largest required lead time of the coded members for the first burst's timestamp
CODED_LEAD_MARGIN_US = 300_000
# a board buffers bursts only up to its stated buffer_capacity, and drops what does not fit: the
# tap sends no further ahead of the clock than this share of what the buffer holds
BUFFER_FILL_SHARE = 0.6
SENDER_INTERVAL_S = 0.02
# bursts demuxed ahead of the timeline, per variant (32 ms each)
_BURST_QUEUE_SIZE = 96

# the one place the role is registered: importing this module is enough
iclforge_role.register()


def pcm_message_bytes(audio_format: SupportedAudioFormat) -> int:
    """
    Return the size of one PCM chunk message of the push stream for a format.

    :param audio_format: The PCM format a client lists.
    """
    samples = audio_format.sample_rate * PCM_CHUNK_US // 1_000_000
    return samples * audio_format.channels * (audio_format.bit_depth // 8) + CHUNK_OVERHEAD_BYTES


def fit_chunk_limit(
    formats: list[SupportedAudioFormat],
    preferred: tuple[SendspinAudioFormat | None, AudioCodec | None],
    max_chunk_bytes: int | None,
) -> tuple[SendspinAudioFormat | None, AudioCodec | None]:
    """
    Return the format to prefer so that no chunk message is larger than the client takes.

    A board that states ``max_chunk_bytes`` below the size of a PCM chunk (the C6 states 4096,
    a 48 kHz 16-bit stereo chunk is 4,829 bytes) closes the connection at stream start. Opus, and
    failing that FLAC, chunks are smaller, so the first of those the client lists is preferred,
    at the sample rate in use. Anything else (a limit that is not stated, a format that already
    fits, a client without a compressed codec) leaves the choice as it was.

    :param formats: The formats the client lists, in its own priority order.
    :param preferred: The (format, codec) MA would otherwise prefer, (None, None) for the client's top.
    :param max_chunk_bytes: The largest message the client states it takes.
    """
    if not max_chunk_bytes or not formats:
        return preferred
    fmt, codec = preferred
    if fmt is None or codec is None:
        top = formats[0]
        codec, rate = top.codec, top.sample_rate
        size = pcm_message_bytes(top)
    else:
        rate = fmt.sample_rate
        size = fmt.sample_rate * PCM_CHUNK_US // 1_000_000 * fmt.channels * (fmt.bit_depth // 8)
        size += CHUNK_OVERHEAD_BYTES
    if codec != AudioCodec.PCM or size <= max_chunk_bytes:
        return preferred
    for wanted in (AudioCodec.OPUS, AudioCodec.FLAC):
        options = [f for f in formats if f.codec == wanted]
        options.sort(key=lambda f: (f.sample_rate != rate, f.channels != 2))
        if options:
            best = options[0]
            return (
                SendspinAudioFormat(
                    sample_rate=best.sample_rate, bit_depth=best.bit_depth, channels=best.channels
                ),
                wanted,
            )
    return preferred


def resume_position_s(seek_s: float, duration_s: float | None) -> float:
    """
    Return where a coded item should start: over from 0 when it is asked to start at its end.

    A queue that stopped or ended on this item can hand over its end as the resume position;
    started there the item has nothing left to send and the play button seems to do nothing.

    :param seek_s: The position the queue asks to start at, in seconds.
    :param duration_s: The duration of the item, if known.
    """
    if duration_s and seek_s >= duration_s - RESTART_THRESHOLD_S:
        return 0.0
    return seek_s


# monotonic time at which each client's coded role was activated by this module
_ACTIVATED_AT: dict[str, float] = {}


def remaining_hold_s(activation_ages_s: Iterable[float], hold_s: float | None = None) -> float:
    """
    Return how much longer the first commit has to wait for the sinks' clocks to converge.

    A sink holds a coded start until its clock converged, about ``hold_s`` after its role was
    activated; a role that has been active that long (a second coded item in a row) needs no hold.

    :param activation_ages_s: Seconds since each coded member's role was activated.
    :param hold_s: The convergence time.
    """
    hold = CODED_START_HOLD_S if hold_s is None else hold_s
    return max((max(0.0, hold - age) for age in activation_ages_s), default=0.0)


def coded_start_lead_us(
    required_lead_us: Iterable[int], margin_us: int = CODED_LEAD_MARGIN_US
) -> int:
    """
    Return how far ahead of now a coded stream starts: the largest required lead plus a margin.

    :param required_lead_us: The required lead time of every coded member, in microseconds.
    :param margin_us: Headroom for send stalls and network jitter.
    """
    return max(required_lead_us, default=0) + margin_us


def first_play_start_us(
    now_us: int,
    required_lead_us: Iterable[int],
    natural_floor_us: int = 0,
    margin_us: int = CODED_LEAD_MARGIN_US,
) -> int:
    """
    Return the timeline timestamp of the group's first commit when it has coded members.

    The PCM members keep at least their own lead (``natural_floor_us``, what the push stream would
    have used); the coded ones decide when theirs is larger.

    :param now_us: The server clock now.
    :param required_lead_us: The required lead time of every coded member.
    :param natural_floor_us: The lead the push stream would use without coded members.
    :param margin_us: Headroom for the coded members.
    """
    return now_us + max(coded_start_lead_us(required_lead_us, margin_us), natural_floor_us)


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


def flush_previous_stream(client: SendspinClient) -> None:
    """
    Have a member drop what its ``player@v1`` stream still holds, before its roles are switched.

    A member with a big buffer (the PC states 33.5 MB, 30 s or more of PCM queued in it) must
    not play on, or sit on the old stream, while the new roles activate. stream/clear goes as a
    priority message: the role queue is discarded when the role is deactivated.

    :param client: The member about to change role.
    """
    if client.role(PLAYER_ROLE_ID) is None:
        return
    client.drop_pending_binary([PLAYER_FAMILY])
    if (connection := getattr(client, "connection", None)) is not None:
        connection.send_priority_message(
            StreamClearMessage(payload=StreamClearPayload(roles=[PLAYER_FAMILY]))
        )


def client_support(client: SendspinClient) -> iclforge_role.IclforgePlayerSupport | None:
    """Return the role support a client stated in its hello, if any."""
    return getattr(client.info, f"{iclforge_role.FAMILY}_support", None)


@dataclass(slots=True)
class _Lane:
    """One variant's burst feed and the members it goes to."""

    variant: CodedVariant
    clients: list[SendspinClient]
    queue: asyncio.Queue[eac3.Burst] = field(
        default_factory=lambda: asyncio.Queue(maxsize=_BURST_QUEUE_SIZE)
    )
    task: asyncio.Task[None] | None = None
    sent: int = 0
    sent_bytes: int = 0
    next_samples: int = 0
    held: eac3.Burst | None = None
    error: str | None = None
    handled: bool = False
    finished: bool = False
    demux_done: bool = False
    bad_clients: list[tuple[SendspinClient, str]] = field(default_factory=list)


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
        stereo_bit_rate: int = STEREO_BIT_RATE,
    ) -> None:
        """Initialize the tap, use :func:`build_coded_tap` instead."""
        self._stereo_bit_rate = stereo_bit_rate
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
        self._clock_anchor: tuple[int, float] | None = None
        self._sender: asyncio.Task[None] | None = None
        self._background: set[asyncio.Task[None]] = set()
        self._expected_members = any(lane.clients for lane in lanes)

    @property
    def sent_nothing(self) -> bool:
        """Return True if members were expected to get bursts and not one was sent."""
        return self._expected_members and all(lane.sent == 0 for lane in self._lanes)

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
        # the demux starts at once: its cold start overlaps the hold instead of following it
        for lane in self._lanes:
            lane.task = asyncio.create_task(self._demux(lane))
            lane.task.add_done_callback(self._log_task_end)
        # a member whose role never reports its state (gone, or stuck) must not hold the group up
        for lane in self._lanes:
            for client in list(lane.clients):
                role = coded_role(client)
                try:
                    if role is None:
                        raise iclforge_role.RoleError("coded role not active")
                    try:
                        await role.wait_for_state(STATE_TIMEOUT_S)
                    except TimeoutError:
                        if not getattr(client, "is_connected", False):
                            raise
                        self._logger.warning(
                            "%s has not reported its coded role yet (still ending its previous "
                            "stream?), waiting up to %.0f s more",
                            client.client_id,
                            STATE_PATIENCE_S,
                        )
                        await role.wait_for_state(STATE_PATIENCE_S)
                except (iclforge_role.RoleError, TimeoutError) as err:
                    self._drop_client(lane, client, f"its coded role is not ready: {err!r}")
        if not any(lane.clients for lane in self._lanes):
            raise iclforge_role.RoleError("no member is ready for the coded stream")
        now = time.monotonic()
        ages = [
            now - _ACTIVATED_AT.get(client.client_id, now - CODED_START_HOLD_S)
            for lane in self._lanes
            for client in lane.clients
        ]
        if (hold := remaining_hold_s(ages)) > 0:
            await asyncio.sleep(hold)
        await self._prime()
        for lane in self._lanes:
            for client in lane.clients:
                if (role := coded_role(client)) is not None:
                    role.start_stream(DATA_TYPE, self._sample_rate)
        self._sender = asyncio.create_task(self._send_loop())

    async def _prime(self) -> None:
        """Wait until every lane has its first bursts ready, so none starts behind its slot."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + PRIME_TIMEOUT_S
        while loop.time() < deadline:
            if all(
                lane.queue.qsize() >= PRIME_BURSTS or lane.demux_done or lane.error is not None
                for lane in self._lanes
            ):
                return
            await asyncio.sleep(0.02)
        self._logger.warning(
            "Coded lanes not primed after %.1f s: %s",
            PRIME_TIMEOUT_S,
            {lane.variant.value: lane.queue.qsize() for lane in self._lanes},
        )

    def _drop_client(self, lane: _Lane, client: SendspinClient, reason: str) -> None:
        """Take one member out of a lane and give it the PCM channel (in the background)."""
        if client not in lane.clients:
            return
        lane.clients.remove(client)
        self._logger.error(
            "Coded %s lane: %s dropped (%s). Switching it to PCM for the rest of the item.",
            lane.variant.value,
            client.client_id,
            reason,
        )
        self._spawn(self._to_pcm([client]))

    def first_play_start_us(self, now_us: int, natural_floor_us: int = 0) -> int:
        """
        Return the play start to give the group's first commit.

        :param now_us: The server clock now.
        :param natural_floor_us: The lead the push stream would use without coded members.
        """
        required = [
            role.get_required_lead_time_us()
            for lane in self._lanes
            for client in lane.clients
            if (role := coded_role(client)) is not None
        ]
        return first_play_start_us(now_us, required, natural_floor_us)

    async def on_commit(self, commit_start_us: int, duration_us: int, now_us: int) -> None:
        """
        Send the bursts that start inside the PCM span just committed.

        :param commit_start_us: Timeline timestamp the committed PCM starts at.
        :param duration_us: Duration of the committed PCM.
        :param now_us: The server clock now, to report the lead of the first burst.
        """
        self._clock_anchor = (now_us, time.monotonic())
        if self._timeline_start_us is None:
            self._timeline_start_us = commit_start_us
            self.first_lead_us = commit_start_us - now_us
            self._logger.info(
                "Coded tap starts at %d us, %.0f ms from now",
                commit_start_us,
                self.first_lead_us / 1000,
            )
        self._produced_us += duration_us
        for lane in self._lanes:
            self._send_due(lane)
            self._handle_failures(lane)

    async def close(self, *, drain: bool = False, flush: bool = True) -> None:
        """
        End the coded streams and stop the demux tasks.

        :param drain: Send what is still queued first (end of the item), else drop it.
        :param flush: Have the sinks drop what they hold at once (stop, pause, new media, seek)
            before the stream ends; False at the natural end of the item, which they play out.
        """
        tasks = [t for t in (self._sender, *(lane.task for lane in self._lanes)) if t is not None]
        tasks.extend(self._background)
        for task in tasks:
            task.cancel()
        for lane in self._lanes:
            if drain:
                self._produced_us = 1 << 62
                with contextlib.suppress(Exception):
                    self._send_due(lane)
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
                    try:
                        if flush:
                            self._flush_sink(client)
                        role.end_stream()
                    except Exception as err:
                        self._logger.warning(
                            "Ending the coded stream of %s failed: %r", client.client_id, err
                        )
        # the sinks hear the end first, and only then is the (possibly slow) ffmpeg teardown awaited
        if tasks:
            # never wait unbounded: a stuck ffmpeg or send must not hold the session's lock
            _, pending = await asyncio.wait(tasks, timeout=CLOSE_TIMEOUT_S)
            if pending:
                self._logger.warning(
                    "%d coded task(s) did not stop within %.1f s", len(pending), CLOSE_TIMEOUT_S
                )

    @staticmethod
    def _flush_sink(client: SendspinClient) -> None:
        """
        Have the sink drop what it holds, now.

        stream/clear goes as a priority message: the role queue it would otherwise wait in is
        discarded when the group's stop deactivates the role, and the clear would never be sent.
        """
        client.drop_pending_binary([iclforge_role.FAMILY])
        if (connection := getattr(client, "connection", None)) is not None:
            connection.send_priority_message(
                StreamClearMessage(payload=StreamClearPayload(roles=[iclforge_role.FAMILY]))
            )

    def _send_due(self, lane: _Lane) -> None:
        """Send queued bursts of a lane whose start lies inside the committed timeline."""
        if self._timeline_start_us is None or lane.error is not None or lane.finished:
            return
        while True:
            burst = lane.held
            lane.held = None
            if burst is None:
                try:
                    burst = lane.queue.get_nowait()
                except asyncio.QueueEmpty:
                    lane.finished = lane.demux_done
                    return
            offset_us = lane.next_samples * 1_000_000 // self._sample_rate
            stamp = self._timeline_start_us + offset_us
            if offset_us >= self._produced_us or stamp - self._now_us() > self._horizon_us(
                lane, len(burst.payload)
            ):
                lane.held = burst
                return
            for client in list(lane.clients):
                role = coded_role(client)
                if role is None:
                    # the member is gone (disconnected or moved on): the lane carries on without it
                    lane.bad_clients.append((client, "its coded role is gone"))
                    lane.clients.remove(client)
                    continue
                if role.stream_started:
                    try:
                        role.send_burst(burst, stamp)
                    except (iclforge_role.RoleError, ConnectionError, OSError) as err:
                        lane.bad_clients.append((client, f"sending failed: {err}"))
                        lane.clients.remove(client)
            lane.next_samples += burst.samples
            lane.sent += 1
            lane.sent_bytes += len(burst.payload)

    def _now_us(self) -> int:
        """Return the server clock now, extrapolated from the last commit."""
        if self._clock_anchor is None:
            return 0
        anchor_us, anchor_s = self._clock_anchor
        return anchor_us + int((time.monotonic() - anchor_s) * 1_000_000)

    def _horizon_us(self, lane: _Lane, payload_bytes: int) -> int:
        """
        Return how far ahead of the clock a lane may send.

        A share of what the smallest buffer among its members holds; unlimited when no member
        states a capacity.
        """
        horizon = 1 << 62
        for client in lane.clients:
            if (role := coded_role(client)) is None or role.support is None:
                continue
            capacity = role.support.buffer_capacity
            bursts = max(1, capacity // (payload_bytes + eac3.BURST_HEADER_BYTES))
            horizon = min(
                horizon, int(bursts * 1_000_000 * 1536 / self._sample_rate * BUFFER_FILL_SHARE)
            )
        return horizon

    async def _send_loop(self) -> None:
        """Send queued bursts as the clock moves, until every lane has run dry."""
        while not all(lane.finished or lane.error is not None for lane in self._lanes):
            for lane in self._lanes:
                self._send_due(lane)
                self._handle_failures(lane)
            await asyncio.sleep(SENDER_INTERVAL_S)

    async def _demux(self, lane: _Lane) -> None:
        """Feed a lane's queue with bursts built from the item's access units."""
        builder = eac3.Eac3BurstBuilder()
        stereo = lane.variant == CodedVariant.STEREO
        stream = get_eac3_demux_stream(
            self._path, self._seek, stereo_bit_rate=self._stereo_bit_rate if stereo else None
        )
        try:
            async for unit in iter_eac3_access_units(_chunks_of(stream)):
                if (burst := builder.push(unit)) is not None:
                    # the commits may all be done by now (a short item is committed in one go),
                    # so the demux also sends what the committed timeline already covers
                    self._send_due(lane)
                    await lane.queue.put(burst)
                    self._send_due(lane)
        except (PassthroughStreamError, eac3.Eac3Error) as err:
            lane.error = f"the {lane.variant.value} stream of {self._path} broke: {err}"
        finally:
            # no await on the queue here: it may be full and the task may be cancelled
            lane.demux_done = True
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await asyncio.wait_for(stream.aclose(), CLOSE_TIMEOUT_S)

    def _handle_failures(self, lane: _Lane) -> None:
        """
        Act on what went wrong in a lane, without ever waiting here (this runs in the commit loop).

        A broken stream moves all of the lane's members to PCM; a member that failed alone
        moves by itself and the others carry on.
        """
        if lane.error is not None and not lane.handled:
            lane.handled = True
            members = list(lane.clients)
            lane.clients = []
            self._logger.error(
                "Coded %s lane failed for %s: %s. Switching them to PCM for the rest of the item.",
                lane.variant.value,
                [c.client_id for c in members],
                lane.error,
            )
            self._spawn(self._to_pcm(members))
        while lane.bad_clients:
            client, reason = lane.bad_clients.pop()
            self._logger.error(
                "Coded %s lane: %s dropped (%s). Switching it to PCM for the rest of the item.",
                lane.variant.value,
                client.client_id,
                reason,
            )
            self._spawn(self._to_pcm([client]))

    def _spawn(self, coro: Coroutine[Any, Any, None]) -> None:
        """Run a recovery in the background, kept so that close() can cancel it."""
        task = asyncio.get_running_loop().create_task(coro)
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def _to_pcm(self, clients: list[SendspinClient]) -> None:
        """
        Give members the PCM channel for the rest of the item.

        They leave coded mode, which re-activates their ``player@v1``; the push stream takes
        them in as late joiners (it replays the PCM it still holds).
        """
        for client in clients:
            if (role := coded_role(client)) is not None:
                with contextlib.suppress(Exception):
                    role.end_stream()
            try:
                await iclforge_role.set_mode(client, "pcm", STATE_TIMEOUT_S)
            except (iclforge_role.RoleError, TimeoutError) as err:
                self._logger.error(
                    "%s could not be switched to PCM (%s), it recovers with the next item",
                    client.client_id,
                    err,
                )
        self.pcm_clients.extend(clients)

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
    if candidate and streamdetails is not None and isinstance(streamdetails.path, str):
        if reason := await probe_eac3_stream(streamdetails.path, seek_position):
            logger.warning(
                "%s is not passed through as a coded stream (%s): every member gets PCM",
                streamdetails.path,
                reason,
            )
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
            was_coded = coded_role(client) is not None
            if variant != CodedVariant.PCM:
                flush_previous_stream(client)
            await iclforge_role.set_mode(client, "pcm" if variant == CodedVariant.PCM else "coded")
            if variant == CodedVariant.PCM:
                _ACTIVATED_AT.pop(client.client_id, None)
            elif not was_coded:
                _ACTIVATED_AT[client.client_id] = time.monotonic()
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
