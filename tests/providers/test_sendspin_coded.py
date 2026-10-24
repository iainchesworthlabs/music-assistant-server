"""Tests for the coded (undecoded) E-AC-3 tap of the Sendspin provider."""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from typing import Any

import pytest

from music_assistant.providers.sendspin import coded
from music_assistant.providers.sendspin.coded import (
    CodedTap,
    CodedVariant,
    _Lane,
    choose_member_variant,
    client_support,
)
from music_assistant.providers.sendspin.ma_role import eac3
from music_assistant.providers.sendspin.ma_role import role as iclforge_role


def _support(**overrides: Any) -> iclforge_role.IclforgePlayerSupport:
    raw: dict[str, Any] = {
        "data_types": ["eac3", "ac3"],
        "sample_rates": [48000],
        "buffer_capacity": 1_000_000,
    }
    raw.update(overrides)
    return iclforge_role.parse_support(raw)


@pytest.mark.parametrize(
    ("support", "expected"),
    [
        (None, CodedVariant.PCM),
        (_support(), CodedVariant.FULL),
        # 0 means unstated (hearth-player), which is no limit
        (_support(max_coded_channels={"eac3": 0}, max_objects={"eac3": 0}), CodedVariant.FULL),
        (_support(max_coded_channels={"eac3": 8}), CodedVariant.FULL),
        (_support(max_coded_channels={"eac3": 2}), CodedVariant.STEREO),
        (_support(max_coded_channels={"eac3": 1}), CodedVariant.PCM),
        # a limit on another data type does not apply to eac3
        (_support(max_coded_channels={"ac3": 2}), CodedVariant.FULL),
        (_support(data_types=["ac3"]), CodedVariant.PCM),
        (_support(sample_rates=[44100]), CodedVariant.PCM),
        # fewer objects than the stream carries still gets the whole stream
        (_support(max_objects={"eac3": 0}), CodedVariant.FULL),
        (_support(max_objects={"eac3": 12}), CodedVariant.FULL),
    ],
)
def test_choose_member_variant(
    support: iclforge_role.IclforgePlayerSupport | None, expected: CodedVariant
) -> None:
    """The variant follows the support the board stated."""
    assert choose_member_variant(support, channels=6, sample_rate=48000) == expected


def test_stereo_source_never_needs_the_stereo_variant() -> None:
    """A two channel stream fits a board that lists two coded channels."""
    support = _support(max_coded_channels={"eac3": 2})
    assert choose_member_variant(support, channels=2, sample_rate=48000) == CodedVariant.FULL


def test_client_support_reads_the_hello_object() -> None:
    """The support object sits on the client info under the role family name."""
    support = _support()
    client = SimpleNamespace(info=SimpleNamespace(_iclforge_player_support=support))
    assert client_support(client) is support  # type: ignore[arg-type]
    assert client_support(SimpleNamespace(info=SimpleNamespace())) is None  # type: ignore[arg-type]


class _FakeRole:
    """Records the bursts a lane sends."""

    stream_started = True

    def __init__(self) -> None:
        self.sent: list[tuple[int, int]] = []

    def send_burst(self, burst: eac3.Burst, timestamp_us: int) -> None:
        self.sent.append((timestamp_us, burst.samples))


def _tap(burst_count: int, monkeypatch: pytest.MonkeyPatch) -> tuple[CodedTap, _FakeRole]:
    role = _FakeRole()
    monkeypatch.setattr(coded, "coded_role", lambda _client: role)
    client = SimpleNamespace(client_id="c1", role=lambda _role_id: role)
    lane = _Lane(variant=CodedVariant.FULL, clients=[client])  # type: ignore[list-item]
    for _ in range(burst_count):
        lane.queue.put_nowait(eac3.Burst(payload=b"\x00" * 4))
    tap = CodedTap(logging.getLogger("test"), "x.ec3", 0, 6, 48000, [lane], [])
    return tap, role


async def test_bursts_follow_the_pcm_timeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """A burst is sent once the committed PCM covers its start, stamped from the timeline start."""
    tap, role = _tap(10, monkeypatch)
    await tap.on_commit(1_000_000, 20_000, 400_000)
    # the first burst starts at offset 0 (inside the first 20 ms), the second at 32 ms (outside)
    assert role.sent == [(1_000_000, 1536)]
    assert tap.first_lead_us == 600_000
    await tap.on_commit(1_020_000, 100_000, 410_000)
    # 120 ms of PCM are committed: bursts at 0, 32, 64 and 96 ms are out, 128 ms is not
    assert [t - 1_000_000 for t, _ in role.sent] == [0, 32_000, 64_000, 96_000]
    await asyncio.sleep(0)


async def test_bursts_are_sent_once_and_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """Many small commits send every burst exactly once with a gapless stamp sequence."""
    tap, role = _tap(40, monkeypatch)
    start = 5_000_000
    committed = 0
    for _ in range(60):
        await tap.on_commit(start + committed, 25_000, 0)
        committed += 25_000
    stamps = [t for t, _ in role.sent]
    assert len(stamps) == 40
    assert stamps == [start + n * 32_000 for n in range(40)]
