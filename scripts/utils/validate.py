"""
validate.py — Pre-processing audio file validation via ffprobe.

Checks that the file actually contains audio before the pipeline
invests time in conversion, transcription, and diarization.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ValidationResult:
    valid: bool                              # True = proceed, False = abort
    errors: list[str] = field(default_factory=list)     # hard blockers
    warnings: list[str] = field(default_factory=list)   # non-fatal concerns

    # ── extracted metadata ────────────────────────────────────────
    codec_name: Optional[str] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    duration_s: Optional[float] = None       # seconds
    bit_rate_bps: Optional[int] = None

    def format(self) -> str:
        """One-line summary suitable for a log message."""
        parts = [f"codec={self.codec_name or '?'}",
                 f"sr={self.sample_rate or '?'}",
                 f"ch={self.channels or '?'}",
                 f"duration={self.duration_s or '?'}s"]
        return ", ".join(parts)


# ── thresholds ─────────────────────────────────────────────────────────

MIN_DURATION_S = 1.0          # reject files shorter than this
WARN_DURATION_S = 3600 * 4    # warn on files longer than 4 hours
MIN_SAMPLE_RATE = 8000
MAX_SAMPLE_RATE = 48000


def validate_audio(path: Path) -> ValidationResult:
    """Validate an audio file before processing.

    Returns a ``ValidationResult``.  If ``valid`` is ``False`` the
    pipeline should abort immediately (the file cannot be processed).
    Warnings are informational only and do not block processing.
    """
    result = ValidationResult(valid=True)

    if not path.exists():
        result.valid = False
        result.errors.append("File not found")
        return result

    if not path.is_file():
        result.valid = False
        result.errors.append("Path is not a regular file")
        return result

    size = path.stat().st_size
    if size == 0:
        result.valid = False
        result.errors.append("File is empty (0 bytes)")
        return result

    # ── probe with ffprobe ──────────────────────────────────────
    try:
        proc = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_format", "-show_streams",
                str(path),
            ],
            capture_output=True, text=True, timeout=30,
        )
    except FileNotFoundError:
        result.valid = False
        result.errors.append("ffprobe not found — is ffmpeg installed?")
        return result
    except subprocess.TimeoutExpired:
        result.valid = False
        result.errors.append("ffprobe timed out — file may be too large or corrupted")
        return result

    if proc.returncode != 0 or not proc.stdout.strip():
        result.valid = False
        result.errors.append("File does not appear to contain valid audio (ffprobe failed)")
        return result

    # ── parse JSON ──────────────────────────────────────────────
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        result.valid = False
        result.errors.append("ffprobe returned unparseable output")
        return result

    streams: list = data.get("streams", [])
    if not streams:
        result.valid = False
        result.errors.append("No streams found in file")
        return result

    # ── look for the first audio stream ─────────────────────────
    audio_stream = None
    for s in streams:
        if s.get("codec_type") == "audio":
            audio_stream = s
            break

    if audio_stream is None:
        result.valid = False
        result.errors.append(f"No audio stream found — file has {len(streams)} non-audio stream(s)")
        return result

    # ── extract metadata ────────────────────────────────────────
    result.codec_name = audio_stream.get("codec_name")
    result.channels = audio_stream.get("channels")
    try:
        result.sample_rate = int(audio_stream["sample_rate"])
    except (KeyError, ValueError, TypeError):
        pass
    try:
        result.duration_s = float(audio_stream["duration"])
    except (KeyError, ValueError, TypeError):
        # fall back to format-level duration
        fmt = data.get("format", {})
        try:
            result.duration_s = float(fmt["duration"])
        except (KeyError, ValueError, TypeError):
            pass
    try:
        result.bit_rate_bps = int(audio_stream.get("bit_rate", 0))
    except (ValueError, TypeError):
        pass

    # ── sanity checks ───────────────────────────────────────────
    if result.duration_s is not None and result.duration_s < MIN_DURATION_S:
        result.valid = False
        result.errors.append(
            f"Duration too short ({result.duration_s:.1f}s — minimum {MIN_DURATION_S}s)"
        )

    if result.duration_s is not None and result.duration_s > WARN_DURATION_S:
        result.warnings.append(
            f"Very long recording ({result.duration_s / 3600:.1f}h) — "
            "processing will take a long time"
        )

    if result.sample_rate is not None:
        if result.sample_rate < MIN_SAMPLE_RATE:
            result.warnings.append(
                f"Low sample rate ({result.sample_rate} Hz) — transcription accuracy may suffer"
            )
        elif result.sample_rate > MAX_SAMPLE_RATE:
            result.warnings.append(
                f"High sample rate ({result.sample_rate} Hz) — "
                "will be resampled, no quality loss"
            )

    return result
