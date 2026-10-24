"""Tests for the start lead, the up-front probe, lane failures and the role-selection guard."""

from __future__ import annotations

import asyncio
import logging
import shutil
from collections.abc import AsyncGenerator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from aiosendspin.server.connection import SendspinConnection
from aiosendspin.server.roles.negotiation import negotiate_roles
from music_assistant_models.enums import ContentType, MediaType, StreamType
from music_assistant_models.media_items import AudioFormat
from music_assistant_models.streamdetails import StreamDetails

from music_assistant.helpers.passthrough import probe_eac3_stream
from music_assistant.providers.sendspin import coded
from music_assistant.providers.sendspin.coded import (
    CODED_LEAD_MARGIN_US,
    CodedTap,
    CodedVariant,
    _Lane,
    build_coded_tap,
    coded_start_lead_us,
    first_play_start_us,
)
from music_assistant.providers.sendspin.ma_role import eac3
from music_assistant.providers.sendspin.ma_role import role as iclforge_role

AUDIO = Path(__file__).parent.parent / "fixtures" / "audio"
EAC3 = AUDIO / "atmos-51-1s.ec3"
AC3 = AUDIO / "tone-ac3-1s.ac3"
needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def test_coded_lead_is_the_largest_required_lead_plus_the_margin() -> None:
    """The coded start allows every member's required lead and the jitter margin on top."""
    assert CODED_LEAD_MARGIN_US == 300_000
    assert coded_start_lead_us([500_000]) == 800_000
    assert coded_start_lead_us([200_000, 900_000, 500_000]) == 1_200_000
    assert coded_start_lead_us([500_000], margin_us=0) == 500_000
    assert coded_start_lead_us([]) == CODED_LEAD_MARGIN_US


def test_first_play_start_keeps_the_pcm_lead_when_it_is_larger() -> None:
    """The group's own lead is a floor: coded members can only add to it."""
    assert first_play_start_us(10_000_000, [500_000]) == 10_800_000
    assert first_play_start_us(10_000_000, [500_000], natural_floor_us=2_000_000) == 12_000_000
    # the old behaviour: a 490 ms natural lead against a 500 ms requirement left no margin
    old_lead = 490_000
    assert old_lead < 500_000 + CODED_LEAD_MARGIN_US
    assert first_play_start_us(0, [500_000], old_lead) - 0 >= 500_000 + CODED_LEAD_MARGIN_US


@needs_ffmpeg
async def test_probe_accepts_eac3_and_refuses_ac3(tmp_path: Path) -> None:
    """E-AC-3 passes the probe; an AC-3 file (not packed into bursts) is turned down."""
    assert await probe_eac3_stream(str(EAC3)) is None
    reason = await probe_eac3_stream(str(AC3))
    assert reason is not None
    assert "E-AC-3" in reason
    assert await probe_eac3_stream(str(tmp_path / "missing.ec3")) is not None


class _Support:
    def __init__(self) -> None:
        self.info = SimpleNamespace(
            _iclforge_player_support=iclforge_role.parse_support(
                {"data_types": ["eac3"], "sample_rates": [48000], "buffer_capacity": 1_000_000}
            )
        )
        self.client_id = "board"


def _details(path: Path, codec: ContentType = ContentType.EAC3) -> StreamDetails:
    return StreamDetails(
        provider="filesystem_local",
        item_id=path.name,
        audio_format=AudioFormat(content_type=codec, channels=6, sample_rate=48000),
        media_type=MediaType.TRACK,
        stream_type=StreamType.LOCAL_FILE,
        path=str(path),
    )


@pytest.fixture
def modes(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Record the mode every set_mode call asked for."""
    calls: list[tuple[str, str]] = []

    async def set_mode(client: Any, mode: str, *_args: Any) -> None:
        calls.append((client.client_id, mode))

    monkeypatch.setattr(iclforge_role, "set_mode", set_mode)
    return calls


@needs_ffmpeg
async def test_unsupported_stream_gives_every_member_pcm_from_the_start(
    modes: list[tuple[str, str]],
) -> None:
    """An AC-3 stream mislabelled as E-AC-3 never reaches a coded lane."""
    tap = await build_coded_tap(logging.getLogger("t"), [_Support()], _details(AC3))  # type: ignore[list-item]
    assert tap is None
    assert modes == [("board", "pcm")]


@needs_ffmpeg
async def test_supported_stream_gets_a_coded_lane(modes: list[tuple[str, str]]) -> None:
    """The same board gets the full stream for real E-AC-3."""
    tap = await build_coded_tap(logging.getLogger("t"), [_Support()], _details(EAC3))  # type: ignore[list-item]
    assert tap is not None
    assert tap.variants == {"board": CodedVariant.FULL.value}
    assert modes == [("board", "coded")]


class _FakeRole:
    stream_started = True
    support = None

    def __init__(self) -> None:
        self.sent = 0
        self.ended = False

    def send_burst(self, burst: eac3.Burst, timestamp_us: int) -> None:
        self.sent += 1

    def end_stream(self) -> None:
        self.ended = True


async def test_failed_lane_moves_its_members_to_pcm_once(
    modes: list[tuple[str, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A demux failure mid-item switches that lane's members to PCM, and only once."""
    role = _FakeRole()
    monkeypatch.setattr(coded, "coded_role", lambda _client: role)
    client = SimpleNamespace(client_id="board")
    lane = _Lane(variant=CodedVariant.FULL, clients=[client])  # type: ignore[list-item]
    for _ in range(3):
        lane.queue.put_nowait(eac3.Burst(payload=b"\x00" * 4))
    lane.error = "the full stream of x.ec3 broke: injected"
    tap = CodedTap(logging.getLogger("t"), "x.ec3", 0, 6, 48000, [lane], [])
    await tap.on_commit(1_000_000, 100_000, 0)
    await tap.on_commit(1_100_000, 100_000, 0)
    assert modes == [("board", "pcm")]
    assert role.ended
    assert role.sent == 0
    assert tap.pcm_clients == [client]
    assert lane.clients == []


async def test_failed_switch_is_logged_and_does_not_raise(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """If the PCM switch itself fails the member recovers at the next item, with an error naming it."""

    async def set_mode(*_args: Any) -> None:
        raise iclforge_role.RoleError("no")

    monkeypatch.setattr(iclforge_role, "set_mode", set_mode)
    monkeypatch.setattr(coded, "coded_role", lambda _client: _FakeRole())
    lane = _Lane(variant=CodedVariant.STEREO, clients=[SimpleNamespace(client_id="c6")])  # type: ignore[list-item]
    lane.error = "sending to c6 failed: boom"
    tap = CodedTap(logging.getLogger("t"), "x.ec3", 0, 6, 48000, [lane], [])
    with caplog.at_level(logging.ERROR):
        await tap.on_commit(0, 100_000, 0)
    text = caplog.text
    assert "c6" in text
    assert "sending to c6 failed: boom" in text
    assert "recovers with the next item" in text


def test_pcm_item_activates_exactly_what_stock_activates() -> None:
    """With the role registered, a board listing both families gets stock MA's roles for PCM."""
    stock_hello = ["player@v1", "controller@v1", "metadata@v1"]
    board_hello = ["_iclforge_player@v1", *stock_hello]
    stock = negotiate_roles(stock_hello)
    negotiated = negotiate_roles(board_hello)
    assert iclforge_role.ROLE_ID in negotiated  # the board really negotiates both families
    assert iclforge_role.select_roles("pcm-board", negotiated) == stock
    # a board that lists only the stock roles is untouched, whatever its mode
    assert iclforge_role.select_roles("old-board", stock) == stock


def test_patched_roles_to_activate_matches_the_stock_property() -> None:
    """The process-wide patch on SendspinConnection returns the stock list for a PCM member."""
    stock = negotiate_roles(["player@v1", "controller@v1", "metadata@v1"])
    both = negotiate_roles(["_iclforge_player@v1", "player@v1", "controller@v1", "metadata@v1"])
    fake = SimpleNamespace(
        _playback_capable=True,
        _negotiated_roles=both,
        _client_id="pcm-board",
        _filter_pairing_roles=lambda roles: list(roles),
    )
    patched = SendspinConnection._roles_to_activate.fget(fake)  # type: ignore[attr-defined]
    assert patched == stock
    # and the playback of a coded item is the only thing that changes it
    iclforge_role._MODES["pcm-board"] = "coded"
    try:
        assert iclforge_role.ROLE_ID in SendspinConnection._roles_to_activate.fget(fake)  # type: ignore[attr-defined]
        assert "player@v1" not in SendspinConnection._roles_to_activate.fget(fake)  # type: ignore[attr-defined]
    finally:
        iclforge_role._MODES.pop("pcm-board")


async def test_demux_sends_what_the_committed_timeline_already_covers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A short item is committed in one go; bursts demuxed after that must still be sent."""

    async def fake_demux(*_args: Any, **_kwargs: Any) -> AsyncGenerator[bytes]:
        data = EAC3.read_bytes()
        for i in range(0, len(data), 4096):
            await asyncio.sleep(0)
            yield data[i : i + 4096]

    role = _FakeRole()
    monkeypatch.setattr(coded, "coded_role", lambda _client: role)
    monkeypatch.setattr(coded, "get_eac3_demux_stream", fake_demux)
    lane = _Lane(variant=CodedVariant.FULL, clients=[SimpleNamespace(client_id="b")])  # type: ignore[list-item]
    tap = CodedTap(logging.getLogger("t"), str(EAC3), 0, 6, 48000, [lane], [])
    # every commit is done before the first burst exists: the queue is empty at that point
    await tap.on_commit(1_000_000, 60_000_000, 0)
    assert role.sent == 0
    await tap._demux(lane)
    expected = len(eac3.build_bursts(EAC3.read_bytes()))
    assert role.sent == expected > 20


async def test_sending_stays_inside_the_buffer_the_board_states(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A board drops what does not fit its buffer, so the tap only sends a share of it ahead."""
    support = iclforge_role.parse_support(
        {"data_types": ["eac3"], "sample_rates": [48000], "buffer_capacity": 261_120}
    )
    role = _FakeRole()
    role.support = support  # type: ignore[assignment]
    monkeypatch.setattr(coded, "coded_role", lambda _client: role)
    lane = _Lane(variant=CodedVariant.FULL, clients=[SimpleNamespace(client_id="kit")])  # type: ignore[list-item]
    for _ in range(90):
        lane.queue.put_nowait(eac3.Burst(payload=b"\0" * 3072))
    tap = CodedTap(logging.getLogger("t"), "x.ec3", 0, 6, 48000, [lane], [])
    # the whole item is committed and the first burst is 0.5 s away
    await tap.on_commit(500_000, 60_000_000, 0)
    horizon_bursts = int(0.6 * 261_120 // (3072 + 17)) + 1
    assert 0 < role.sent <= horizon_bursts
    assert role.sent < 90
    # the clock moves on and the rest may go out
    tap._clock_anchor = (2_000_000, tap._clock_anchor[1])  # type: ignore[index]
    tap._send_due(lane)
    assert role.sent == 90
