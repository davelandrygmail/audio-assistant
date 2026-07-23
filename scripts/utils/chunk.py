"""
chunk.py — Split long audio files into overlapping chunks and merge results.

The primary bottleneck in the pipeline is Whisper transcription on CPU.
For recordings longer than ~20 minutes, splitting into shorter chunks:

- Reduces peak memory during transcription
- Enables progress reporting per chunk
- Makes the pipeline resumable (partial results survive mid-run failures)
- Is a prerequisite for future parallel transcription

The overlap between chunks prevents word cutoff at boundaries.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple


def get_wav_duration(wav_path: Path) -> float:
    """Return the duration of a WAV file in seconds via ffprobe."""
    proc = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format", "-show_streams",
            str(wav_path),
        ],
        capture_output=True, text=True, timeout=30,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {wav_path.name}: {proc.stderr.strip()}")

    data = json.loads(proc.stdout)
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "audio":
            dur = stream.get("duration")
            if dur is not None:
                return float(dur)
    fmt = data.get("format", {})
    dur = fmt.get("duration")
    if dur is not None:
        return float(dur)
    raise RuntimeError(f"Could not determine duration of {wav_path.name}")


def split_wav(
    wav_path: Path,
    chunk_secs: float,
    overlap_secs: float = 5.0,
    temp_dir: Path | None = None,
) -> List[Tuple[float, Path]]:
    """Split a WAV file into overlapping chunks via ffmpeg.

    Parameters
    ----------
    wav_path : Path
        Path to a 16 kHz mono WAV.
    chunk_secs : float
        Target chunk duration in seconds (excluding overlap).
    overlap_secs : float
        Seconds of overlap added at the end of each chunk except the last.
    temp_dir : Path or None
        Directory for temporary chunk files.  Uses system temp if ``None``.

    Returns
    -------
    list[tuple[float, Path]]
        Each tuple is ``(start_time_seconds, chunk_wav_path)``.
        Start times are relative to the beginning of the original file.
    """
    duration = get_wav_duration(wav_path)
    if duration <= chunk_secs:
        return [(0.0, wav_path)]

    temp_dir = temp_dir or Path(tempfile.mkdtemp(prefix="chunks_"))
    temp_dir.mkdir(parents=True, exist_ok=True)

    chunks: List[Tuple[float, Path]] = []
    offset = 0.0
    chunk_idx = 0

    while offset < duration:
        chunk_path = temp_dir / f"chunk_{chunk_idx:03d}.wav"

        # Last chunk: just grab the remainder, no overlap needed
        remaining = duration - offset
        if remaining <= chunk_secs + overlap_secs:
            grab = remaining
        else:
            grab = chunk_secs + overlap_secs

        subprocess.run(
            [
                "ffmpeg", "-y", "-ss", str(offset),
                "-i", str(wav_path),
                "-t", str(grab),
                "-acodec", "copy",          # no re-encode — fast
                str(chunk_path),
            ],
            capture_output=True, check=True,
        )

        chunks.append((offset, chunk_path))
        offset += chunk_secs
        chunk_idx += 1

    return chunks


def merge_chunk_transcripts(
    chunk_results: List[Tuple[float, Dict]],
    chunk_secs: float,
    overlap_secs: float = 5.0,
) -> Dict:
    """Merge per-chunk transcription results into one whisper-format dict.

    Each chunk was transcribed independently.  Timestamps are shifted by
    the chunk's start offset, and words in the overlap zone (the last
    *overlap_secs* of each non-final chunk) are discarded to avoid
    duplicate text at boundaries.

    Parameters
    ----------
    chunk_results : list[tuple[float, dict]]
        ``(start_offset, whisper_result_dict)`` from each chunk.
    chunk_secs : float
        The non-overlap chunk duration used when splitting.
    overlap_secs : float
        The overlap duration.

    Returns
    -------
    dict
        Merged whisper-format: ``{"text": ..., "segments": ..., "words": ...}``.
    """
    if not chunk_results:
        return {"text": "", "segments": [], "words": []}

    merged_words: List[Dict] = []
    merged_segments: List[Dict] = []
    merged_text_parts: List[str] = []

    # Total number of chunks that had overlap padding (all except the last)
    n_chunks = len(chunk_results)

    for i, (start_offset, result) in enumerate(chunk_results):
        is_last = (i == n_chunks - 1)

        # ── words ─────────────────────────────────────────────────
        chunk_words = result.get("words", [])
        for w in chunk_words:
            shifted_start = w["start"] + start_offset
            shifted_end = w["end"] + start_offset

            # Discard words in the overlap zone for non-final chunks
            if not is_last:
                word_local_time = w["start"]  # before offset shift
                if word_local_time >= chunk_secs:
                    continue  # this word is in the overlap padding

            merged_words.append({
                "word": w["word"],
                "start": shifted_start,
                "end": shifted_end,
            })

        # ── segments ──────────────────────────────────────────────
        for seg in result.get("segments", []):
            seg_start = seg["start"] + start_offset
            seg_end = seg["end"] + start_offset

            # Discard segments entirely in the overlap padding
            if not is_last and seg["start"] >= chunk_secs:
                continue
            # Clip segment end if it extends into overlap
            if not is_last and seg["end"] > chunk_secs:
                seg_end = start_offset + chunk_secs

            seg_words = []
            if "words" in seg and seg["words"]:
                for sw in seg["words"]:
                    sw_local = sw["start"]
                    if not is_last and sw_local >= chunk_secs:
                        continue
                    seg_words.append({
                        "word": sw["word"],
                        "start": sw["start"] + start_offset,
                        "end": sw["end"] + start_offset,
                    })

            merged_segments.append({
                "start": seg_start,
                "end": seg_end,
                "text": seg.get("text", ""),
                "words": seg_words,
            })

        # ── text ──────────────────────────────────────────────────
        if not is_last and result.get("text"):
            # For non-final chunks, trim overlap from the text end.
            # We can't easily slice text by time, so we just include
            # the full chunk text — word-level dedup handled above.
            pass
        merged_text_parts.append(result.get("text", ""))

    return {
        "text": " ".join(merged_text_parts),
        "segments": merged_segments,
        "words": merged_words,
    }
