"""Tests for music_assistant.helpers.passthrough and the coded queue item stream."""

from __future__ import annotations

import shutil
from collections.abc import AsyncGenerator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from music_assistant_models.enums import ContentType, MediaType, StreamType
from music_assistant_models.errors import AudioError
from music_assistant_models.media_items import AudioFormat
from music_assistant_models.streamdetails import StreamDetails

from music_assistant.controllers.streams.audio import StreamsAudio
from music_assistant.helpers.passthrough import (
    EAC3_SAMPLES_PER_FRAME,
    PassthroughStreamError,
    get_eac3_demux_stream,
    is_coded_passthrough_candidate,
    iter_eac3_access_units,
    parse_eac3_syncframe_header,
    split_eac3_access_units,
)
from music_assistant.providers.filesystem_local import LocalFileSystemProvider

FIXTURE = Path(__file__).parent.parent / "fixtures" / "audio" / "atmos-51-1s.ec3"
needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def _frame(size: int, *, dependent: bool = False) -> bytes:
    """Build a syncframe of ``size`` bytes (even) with a valid E-AC-3 header."""
    frmsiz = size // 2 - 1
    strmtyp = 1 if dependent else 0
    header = bytes(
        [0x0B, 0x77, (strmtyp << 6) | (frmsiz >> 8), frmsiz & 0xFF, 0x3F]  # fscod 0, numblkscod 3
    )
    return header + bytes(size - len(header))


async def _chunks(data: bytes, size: int) -> AsyncGenerator[bytes]:
    for i in range(0, len(data), size):
        yield data[i : i + size]


def _streamdetails(**kwargs: object) -> StreamDetails:
    values: dict[str, object] = {
        "provider": "filesystem_local",
        "item_id": "x.ec3",
        "audio_format": AudioFormat(content_type=ContentType.EAC3),
        "media_type": MediaType.TRACK,
        "stream_type": StreamType.LOCAL_FILE,
        "path": str(FIXTURE),
    }
    values.update(kwargs)
    return StreamDetails(**values)  # type: ignore[arg-type]


def test_parse_header_reports_size_and_dependency() -> None:
    """The frame size is (frmsiz + 1) * 2 and strmtyp 1 marks a dependent substream."""
    assert parse_eac3_syncframe_header(_frame(768)) == (768, False)
    assert parse_eac3_syncframe_header(_frame(320, dependent=True)) == (320, True)


def test_parse_header_rejects_lost_sync() -> None:
    """A header without the syncword is an error, not a guess."""
    with pytest.raises(PassthroughStreamError):
        parse_eac3_syncframe_header(b"\x00\x00\x00\x00\x00")


def test_split_groups_dependent_frames_with_their_independent_frame() -> None:
    """An access unit is an independent frame plus the dependent frames after it."""
    unit_a = _frame(768) + _frame(320, dependent=True)
    unit_b = _frame(768)
    unit_c = _frame(512)
    units, rest = split_eac3_access_units(unit_a + unit_b + unit_c)
    assert units == [unit_a, unit_b]
    assert rest == unit_c
    units, rest = split_eac3_access_units(unit_a + unit_b + unit_c, final=True)
    assert units == [unit_a, unit_b, unit_c]
    assert rest == b""


def test_split_keeps_partial_frame_in_remainder() -> None:
    """A frame that has not fully arrived is never emitted."""
    data = _frame(768) + _frame(768)
    units, rest = split_eac3_access_units(data[:-10])
    assert units == []
    assert rest == data[:-10]


async def test_iter_access_units_independent_of_input_chunking() -> None:
    """Re-chunking yields the same access units however the bytes arrive."""
    units = [_frame(768) + _frame(320, dependent=True), _frame(768), _frame(640)]
    data = b"".join(units)
    for size in (1, 7, 100, len(data)):
        assert [u async for u in iter_eac3_access_units(_chunks(data, size))] == units


async def test_iter_access_units_rejects_truncated_stream() -> None:
    """A stream cut inside a frame raises instead of passing on a broken frame."""
    data = _frame(768) + _frame(768)
    with pytest.raises(PassthroughStreamError):
        _ = [u async for u in iter_eac3_access_units(_chunks(data[:-4], 50))]


def test_candidate_requires_local_eac3_file() -> None:
    """Only a local file with an E-AC-3 content or codec type is a candidate."""
    assert is_coded_passthrough_candidate(_streamdetails())
    in_mp4 = AudioFormat(content_type=ContentType.M4A, codec_type=ContentType.EAC3)
    assert is_coded_passthrough_candidate(_streamdetails(audio_format=in_mp4))
    flac = AudioFormat(content_type=ContentType.FLAC)
    assert not is_coded_passthrough_candidate(_streamdetails(audio_format=flac))
    assert not is_coded_passthrough_candidate(
        _streamdetails(stream_type=StreamType.HTTP, path=None)
    )


def test_raw_ec3_file_gets_eac3_content_type() -> None:
    """Ffprobe never reports a container for a raw .ec3 file, the extension decides."""
    tags = SimpleNamespace(format="eac3")
    ec3 = SimpleNamespace(ext="ec3")
    m4a = SimpleNamespace(ext="m4a")
    assert LocalFileSystemProvider._content_type_for_file(ec3, tags) == ContentType.EAC3  # type: ignore[arg-type]
    assert LocalFileSystemProvider._content_type_for_file(m4a, tags) == ContentType.M4A  # type: ignore[arg-type]


@needs_ffmpeg
async def test_demux_returns_the_source_bitstream_in_whole_access_units() -> None:
    """The demux is lossless: the output equals the raw source, cut on access units."""
    source = FIXTURE.read_bytes()
    chunks = [c async for c in get_eac3_demux_stream(str(FIXTURE), access_units_per_chunk=4)]
    assert b"".join(chunks) == source
    for chunk in chunks:
        units, rest = split_eac3_access_units(chunk, final=True)
        assert rest == b""
        assert b"".join(units) == chunk
    assert len(chunks) > 1
    frames = sum(len(split_eac3_access_units(c, final=True)[0]) for c in chunks)
    assert abs(frames * EAC3_SAMPLES_PER_FRAME - 48000) < 2 * EAC3_SAMPLES_PER_FRAME


@needs_ffmpeg
async def test_demux_seek_skips_the_start() -> None:
    """Seeking starts later in the stream and still begins on an access unit."""
    source = FIXTURE.read_bytes()
    data = b"".join([c async for c in get_eac3_demux_stream(str(FIXTURE), seek_position=0.5)])
    assert 0 < len(data) < len(source)
    assert source.endswith(data)
    assert split_eac3_access_units(data, final=True)[1] == b""


@needs_ffmpeg
async def test_queue_item_passthrough_stream_demuxes_the_file() -> None:
    """The queue item stream yields the coded bytes without touching the PCM machinery."""
    queue_item = MagicMock()
    queue_item.streamdetails = _streamdetails()
    fake_self = SimpleNamespace(logger=MagicMock())
    data = b"".join(
        [c async for c in StreamsAudio.get_queue_item_passthrough_stream(fake_self, queue_item)]  # type: ignore[arg-type]
    )
    assert data == FIXTURE.read_bytes()


async def test_queue_item_passthrough_stream_refuses_pcm_items() -> None:
    """A non coded item cannot be passed through."""
    queue_item = MagicMock()
    queue_item.streamdetails = _streamdetails(
        audio_format=AudioFormat(content_type=ContentType.FLAC)
    )
    fake_self = SimpleNamespace(logger=MagicMock())
    with pytest.raises(AudioError):
        async for _ in StreamsAudio.get_queue_item_passthrough_stream(fake_self, queue_item):  # type: ignore[arg-type]
            pass
