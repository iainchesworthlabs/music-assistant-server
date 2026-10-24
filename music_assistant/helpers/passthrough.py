"""Helpers for passing a coded (undecoded) E-AC-3 stream through to the players."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

from music_assistant_models.enums import ContentType, StreamType

from music_assistant.helpers.process import AsyncProcess

if TYPE_CHECKING:
    from music_assistant_models.streamdetails import StreamDetails

CODED_PASSTHROUGH_CODECS = frozenset({ContentType.EAC3})
EAC3_SYNCWORD = b"\x0b\x77"
EAC3_SAMPLES_PER_BLOCK = 256
# one E-AC-3 syncframe always carries 6 blocks of 256 samples
EAC3_SAMPLES_PER_FRAME = 6 * EAC3_SAMPLES_PER_BLOCK
EAC3_SAMPLE_RATES = (48000, 44100, 32000)
_EAC3_HEADER_SIZE = 5
_EAC3_STRMTYP_DEPENDENT = 1
_EAC3_BSID = 16
_EAC3_FSCOD_RESERVED = 3
# the number of blocks per syncframe is coded as numblkscod unless fscod is reserved
_EAC3_BLOCKS_BY_NUMBLKSCOD = (1, 2, 3, 6)
_EAC3_DEMUX_CHUNK_ACCESS_UNITS = 8


class PassthroughStreamError(Exception):
    """The coded stream is not a well formed E-AC-3 elementary stream."""


def is_coded_passthrough_candidate(streamdetails: StreamDetails) -> bool:
    """
    Return True if the item can be sent to capable players as a coded stream.

    TODO: the role plug-in decides per player group whether this is used. For now
    this only says that the source is a local E-AC-3 file that can be demuxed.

    :param streamdetails: The stream details of the queue item.
    """
    audio_format = streamdetails.audio_format
    if streamdetails.stream_type != StreamType.LOCAL_FILE or not streamdetails.path:
        return False
    return bool({audio_format.content_type, audio_format.codec_type} & CODED_PASSTHROUGH_CODECS)


def parse_eac3_syncframe_header(data: bytes) -> tuple[int, bool]:
    """
    Parse the start of an E-AC-3 syncframe.

    Returns the frame size in bytes and whether the frame belongs to a dependent substream.

    :param data: At least the first 5 bytes of a syncframe.
    :raises PassthroughStreamError: If the data is not an E-AC-3 syncframe.
    """
    if len(data) < _EAC3_HEADER_SIZE or data[:2] != EAC3_SYNCWORD:
        raise PassthroughStreamError("Lost E-AC-3 syncword")
    # syncword(16) strmtyp(2) substreamid(3) frmsiz(11) fscod(2) ... bsid(5)
    strmtyp = data[2] >> 6
    frmsiz = ((data[2] & 0x07) << 8) | data[3]
    if strmtyp == 3:
        raise PassthroughStreamError("Reserved E-AC-3 stream type")
    if data[4] >> 6 == _EAC3_FSCOD_RESERVED and (data[4] >> 4) & 0x03 == _EAC3_FSCOD_RESERVED:
        raise PassthroughStreamError("Reserved E-AC-3 sample rate code")
    return (frmsiz + 1) * 2, strmtyp == _EAC3_STRMTYP_DEPENDENT


def split_eac3_access_units(data: bytes, *, final: bool = False) -> tuple[list[bytes], bytes]:
    """
    Split an E-AC-3 elementary stream into access units.

    An access unit is an independent syncframe plus the dependent syncframes that
    follow it. A unit is only complete once the next independent frame has started,
    so the tail stays in the remainder unless ``final`` is set.

    :param data: Buffered stream bytes starting on a syncframe.
    :param final: True when no more data will follow.
    :return: The complete access units and the unconsumed remainder.
    """
    units: list[bytes] = []
    unit_start = 0
    pos = 0
    while pos + _EAC3_HEADER_SIZE <= len(data):
        frame_size, dependent = parse_eac3_syncframe_header(data[pos : pos + _EAC3_HEADER_SIZE])
        if pos + frame_size > len(data):
            break
        if not dependent and pos > unit_start:
            units.append(data[unit_start:pos])
            unit_start = pos
        pos += frame_size
    if final and pos == len(data) and pos > unit_start:
        units.append(data[unit_start:pos])
        unit_start = pos
    return units, data[unit_start:]


async def iter_eac3_access_units(chunks: AsyncGenerator[bytes]) -> AsyncGenerator[bytes]:
    """
    Re-chunk a raw E-AC-3 byte stream into whole access units.

    :param chunks: Raw elementary stream bytes in arbitrary pieces.
    :raises PassthroughStreamError: If the stream loses syncframe alignment.
    """
    buffer = b""
    async for chunk in chunks:
        buffer += chunk
        units, buffer = split_eac3_access_units(buffer)
        for unit in units:
            yield unit
    units, buffer = split_eac3_access_units(buffer, final=True)
    for unit in units:
        yield unit
    if buffer:
        raise PassthroughStreamError("Truncated E-AC-3 syncframe at end of stream")


async def get_eac3_demux_stream(
    path: str,
    seek_position: float = 0,
    access_units_per_chunk: int = _EAC3_DEMUX_CHUNK_ACCESS_UNITS,
) -> AsyncGenerator[bytes]:
    """
    Demux the audio of a file to a raw E-AC-3 stream without decoding it.

    There are deliberately no filters: no resampling, normalization, fades or
    buffering, the bitstream (including any Atmos extension data) stays untouched.
    Every yielded chunk holds whole access units.

    :param path: Local file with an E-AC-3 audio stream (raw .ec3 or MP4).
    :param seek_position: Start position in seconds.
    :param access_units_per_chunk: Access units per yielded chunk.
    """
    args = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin"]
    if seek_position:
        args += ["-ss", f"{seek_position:.3f}"]
    args += ["-i", path, "-vn", "-dn", "-sn", "-map", "0:a:0", "-c:a", "copy", "-f", "eac3"]
    args += ["pipe:1"]
    async with AsyncProcess(args, stdout=True, stderr=False, name="ffmpeg-demux") as proc:
        pending: list[bytes] = []
        async for unit in iter_eac3_access_units(proc.iter_any()):
            pending.append(unit)
            if len(pending) >= access_units_per_chunk:
                yield b"".join(pending)
                pending = []
        if pending:
            yield b"".join(pending)
