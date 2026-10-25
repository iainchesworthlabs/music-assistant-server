"""Tests for the start lead, the up-front probe, lane failures and the role-selection guard."""

from __future__ import annotations

import asyncio
import logging
import shutil
import time
from collections.abc import AsyncGenerator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from aiosendspin.models import AudioCodec
from aiosendspin.models.player import SupportedAudioFormat
from aiosendspin.server.audio import AudioFormat as SendspinAudioFormat
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
    fit_chunk_limit,
    flush_previous_stream,
    pcm_message_bytes,
    remaining_hold_s,
    resume_position_s,
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

    def role(self, _role_id: str) -> None:
        return None


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
        self.calls: list[str] = []

    def send_burst(self, burst: eac3.Burst, timestamp_us: int) -> None:
        self.sent += 1

    def end_stream(self) -> None:
        self.ended = True
        self.calls.append("end")

    def clear_stream(self) -> None:
        self.calls.append("clear")


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
    await asyncio.sleep(0.05)  # the switch runs in the background
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
        await asyncio.sleep(0.05)
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


def _formats() -> list[Any]:

    return [
        SupportedAudioFormat(codec=AudioCodec.PCM, channels=2, sample_rate=48000, bit_depth=16),
        SupportedAudioFormat(codec=AudioCodec.PCM, channels=2, sample_rate=48000, bit_depth=24),
        SupportedAudioFormat(codec=AudioCodec.OPUS, channels=2, sample_rate=48000, bit_depth=16),
    ]


def test_pcm_chunk_that_does_not_fit_the_board_gives_opus() -> None:
    """A board stating max_chunk_bytes 4096 cannot take 4,829-byte PCM chunks: Opus instead."""
    formats = _formats()
    assert pcm_message_bytes(formats[0]) == 4840  # 25 ms 48 kHz 16-bit stereo + overhead
    fmt, codec = fit_chunk_limit(formats, (None, None), 4096)
    assert codec == AudioCodec.OPUS
    assert fmt is not None
    assert (fmt.sample_rate, fmt.channels) == (48000, 2)


def test_chunk_limit_leaves_the_choice_alone_when_it_fits_or_is_not_stated() -> None:
    """No limit, a roomy limit, or an explicit non-PCM choice is not touched."""
    formats = _formats()
    assert fit_chunk_limit(formats, (None, None), None) == (None, None)
    assert fit_chunk_limit(formats, (None, None), 25_600) == (None, None)
    opus = (SendspinAudioFormat(sample_rate=48000, bit_depth=16, channels=2), AudioCodec.OPUS)
    assert fit_chunk_limit(formats, opus, 4096) == opus


def test_chunk_limit_falls_back_to_flac_then_to_the_original() -> None:
    """Without Opus the first FLAC format is used; without either nothing changes."""
    pcm = _formats()[:2]
    flac = [
        *pcm,
        SupportedAudioFormat(codec=AudioCodec.FLAC, channels=2, sample_rate=48000, bit_depth=16),
    ]
    assert fit_chunk_limit(flac, (None, None), 4096)[1] == AudioCodec.FLAC
    assert fit_chunk_limit(pcm, (None, None), 4096) == (None, None)


def test_hold_is_only_the_part_of_the_convergence_time_still_to_go() -> None:
    """A role that has been active long enough needs no hold; a fresh one the whole of it."""
    assert remaining_hold_s([0.0]) == 2.5
    assert remaining_hold_s([1.0, 10.0]) == 1.5
    assert remaining_hold_s([3.0, 60.0]) == 0.0
    assert remaining_hold_s([]) == 0.0


async def test_cancelling_a_full_demux_never_blocks_the_teardown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The demux task of a full queue is cancelled by a stop; close() must come back at once."""

    async def endless(*_args: Any, **_kwargs: Any) -> AsyncGenerator[bytes]:
        data = EAC3.read_bytes()
        while True:
            for i in range(0, len(data), 4096):
                await asyncio.sleep(0)
                yield data[i : i + 4096]

    role = _FakeRole()
    monkeypatch.setattr(coded, "coded_role", lambda _client: role)
    monkeypatch.setattr(coded, "get_eac3_demux_stream", endless)
    lane = _Lane(variant=CodedVariant.FULL, clients=[SimpleNamespace(client_id="b")])  # type: ignore[list-item]
    tap = CodedTap(logging.getLogger("t"), str(EAC3), 0, 6, 48000, [lane], [])
    lane.task = asyncio.create_task(tap._demux(lane))
    for _ in range(200):
        await asyncio.sleep(0.01)
        if lane.queue.full():
            break
    assert lane.queue.full()
    started = asyncio.get_running_loop().time()
    async with asyncio.timeout(3):
        await tap.close()
    assert asyncio.get_running_loop().time() - started < 1.0
    assert lane.demux_done


async def test_prepare_cancelled_during_the_hold_stops_everything(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stop during the hold cancels prepare(); the demux tasks are then closed in no time."""

    async def endless(*_args: Any, **_kwargs: Any) -> AsyncGenerator[bytes]:
        data = EAC3.read_bytes()
        while True:
            await asyncio.sleep(0)
            yield data[:4096]

    class _Role(_FakeRole):
        async def wait_for_state(self, timeout: float = 0) -> None:
            return None

    role = _Role()
    monkeypatch.setattr(coded, "coded_role", lambda _client: role)
    monkeypatch.setattr(coded, "get_eac3_demux_stream", endless)
    lane = _Lane(variant=CodedVariant.FULL, clients=[SimpleNamespace(client_id="b")])  # type: ignore[list-item]
    tap = CodedTap(logging.getLogger("t"), str(EAC3), 0, 6, 48000, [lane], [])
    coded._ACTIVATED_AT["b"] = time.monotonic()
    task = asyncio.create_task(tap.prepare())
    await asyncio.sleep(0.2)  # inside the 2.5 s hold, the demux is running
    assert not task.done()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    async with asyncio.timeout(2):
        await tap.close()
    assert lane.demux_done
    assert lane.task is not None
    assert lane.task.done()
    coded._ACTIVATED_AT.pop("b", None)


async def test_one_member_failing_does_not_take_the_others_off_the_coded_stream(
    modes: list[tuple[str, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A killed member (send error, or its role gone) alone moves to PCM; the rest carry on."""
    good, bad = _FakeRole(), _FakeRole()

    def failing(*_args: Any) -> None:
        raise iclforge_role.RoleError("socket closed")

    bad.send_burst = failing  # type: ignore[method-assign,assignment]
    roles = {"good": good, "bad": bad}
    monkeypatch.setattr(coded, "coded_role", lambda client: roles.get(client.client_id))
    clients = [SimpleNamespace(client_id=n) for n in ("good", "bad", "gone")]
    lane = _Lane(variant=CodedVariant.FULL, clients=clients)  # type: ignore[arg-type]
    for _ in range(5):
        lane.queue.put_nowait(eac3.Burst(payload=b"\0" * 4))
    tap = CodedTap(logging.getLogger("t"), "x.ec3", 0, 6, 48000, [lane], [])
    await tap.on_commit(1_000_000, 60_000_000, 0)
    await asyncio.sleep(0.05)
    assert good.sent == 5
    assert [c.client_id for c in lane.clients] == ["good"]
    assert sorted(m for m, _ in modes) == ["bad", "gone"]
    assert all(mode == "pcm" for _, mode in modes)
    assert lane.error is None


def test_a_coded_item_asked_to_start_at_its_end_starts_over() -> None:
    """A queue that parked at the end of an item must not start it there (nothing would play)."""
    assert resume_position_s(72.0, 72.0) == 0.0
    assert resume_position_s(71.0, 72.0) == 0.0
    assert resume_position_s(69.0, 72.0) == 69.0
    assert resume_position_s(5.0, 72.0) == 5.0
    assert resume_position_s(12.0, None) == 12.0


def _nothing_sent(tap: CodedTap) -> bool:
    return tap.sent_nothing


async def test_a_tap_that_sent_no_burst_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    """Members expected, nothing sent: reported so the session can log it and not stay 'playing'."""
    monkeypatch.setattr(coded, "coded_role", lambda _client: _FakeRole())
    lane = _Lane(variant=CodedVariant.FULL, clients=[SimpleNamespace(client_id="b")])  # type: ignore[list-item]
    tap = CodedTap(logging.getLogger("t"), "x.ec3", 0, 6, 48000, [lane], [])
    assert tap.sent_nothing
    lane.queue.put_nowait(eac3.Burst(payload=b"\0" * 4))
    await tap.on_commit(1_000_000, 60_000_000, 0)
    assert not tap.sent_nothing


@pytest.mark.parametrize(("flush", "expected"), [(True, ["clear", "end"]), (False, ["end"])])
async def test_stop_flushes_the_sinks_before_ending_and_a_natural_end_does_not(
    monkeypatch: pytest.MonkeyPatch, flush: bool, expected: list[str]
) -> None:
    """Stop, pause, new media and seek flush the sink (stream/clear, as a priority message)."""
    events: list[str] = []
    role = _FakeRole()
    role.end_stream = lambda: events.append("end")  # type: ignore[method-assign]
    monkeypatch.setattr(coded, "coded_role", lambda _client: role)
    connection = SimpleNamespace(
        send_priority_message=lambda message: events.append(
            "clear" if type(message).__name__ == "StreamClearMessage" else "other"
        )
    )
    client = SimpleNamespace(
        client_id="b",
        connection=connection,
        drop_pending_binary=lambda _roles: events.append("drop"),
    )
    lane = _Lane(variant=CodedVariant.FULL, clients=[client])  # type: ignore[list-item]
    tap = CodedTap(logging.getLogger("t"), "x.ec3", 0, 6, 48000, [lane], [])
    await tap.close(flush=flush)
    assert [e for e in events if e != "drop"] == expected


def test_a_big_buffer_member_is_flushed_before_its_roles_switch() -> None:
    """The old player@v1 stream is cleared (priority message) before the coded role activates."""
    events: list[str] = []
    connection = SimpleNamespace(
        send_priority_message=lambda message: events.append(type(message).__name__)
    )
    playing = SimpleNamespace(
        client_id="pc",
        connection=connection,
        role=lambda role_id: object() if role_id == "player@v1" else None,
        drop_pending_binary=lambda roles: events.append(f"drop {roles}"),
    )
    flush_previous_stream(playing)  # type: ignore[arg-type]
    assert events == ["drop ['player']", "StreamClearMessage"]
    events.clear()
    idle = SimpleNamespace(
        client_id="pc",
        connection=connection,
        role=lambda _role_id: None,
        drop_pending_binary=lambda _roles: events.append("drop"),
    )
    flush_previous_stream(idle)  # type: ignore[arg-type]
    assert events == []  # a member without an active player@v1 is left alone


async def test_a_connected_member_ending_its_old_stream_is_waited_for(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A role state that comes late from a connected member does not drop it to PCM."""
    monkeypatch.setattr(coded, "STATE_TIMEOUT_S", 0.05)
    calls: list[float] = []

    class _Slow(_FakeRole):
        def start_stream(self, *_args: Any) -> None:
            return None

        async def wait_for_state(self, timeout: float = 0) -> None:
            calls.append(timeout)
            if len(calls) == 1:
                raise TimeoutError

    role = _Slow()
    monkeypatch.setattr(coded, "coded_role", lambda _client: role)
    monkeypatch.setattr(coded, "get_eac3_demux_stream", _empty_demux)
    client = SimpleNamespace(client_id="pc", is_connected=True)
    lane = _Lane(variant=CodedVariant.FULL, clients=[client])  # type: ignore[list-item]
    tap = CodedTap(logging.getLogger("t"), "x.ec3", 0, 6, 48000, [lane], [])
    coded._ACTIVATED_AT["pc"] = time.monotonic() - 10
    await tap.prepare()
    coded._ACTIVATED_AT.pop("pc", None)
    assert len(calls) == 2
    assert lane.clients == [client]
    await tap.close()


async def _empty_demux(*_args: Any, **_kwargs: Any) -> AsyncGenerator[bytes]:
    empty: list[bytes] = []
    for chunk in empty:
        yield chunk
