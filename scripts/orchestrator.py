"""
orchestrator.py — Central entrypoint for the audio processing pipeline.

Call :func:`process_audio_file` to run the full chain on a single file:
    convert → transcribe → diarize → align → analyze + save outputs.

This module extracts the pipeline logic from the watchdog so it can be
invoked directly (CLI, one-off, cron, test) or driven by the folder watcher.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from scripts.utils.config import get_config
from scripts.processing.convert_wav import to_wav
from scripts.processing.transcribe import transcribe
from scripts.processing.diarize import diarize
from scripts.utils.align import words_from_whisper, assign_speakers, aggregate_utterances
from scripts.ai.llm_analysis import analyze_with_llm


# ── Status tracking ────────────────────────────────────────────────────

_STATUS_FILE = Path(__file__).resolve().parent.parent / "status.json"


def write_status(file: str | None = None, phase: str | None = None,
                 status: str = "idle", error: str | None = None) -> None:
    """Write a lightweight JSON status file so external tools can monitor progress."""
    data: dict = {
        "status": status,        # "idle" | "processing" | "error"
        "file": file,
        "phase": phase,          # "converting" | "transcribing" | "diarizing" | "aligning" | "analyzing" | "archiving"
        "pid": os.getpid(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if error:
        data["error"] = error
    try:
        _STATUS_FILE.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    except OSError:
        pass  # best-effort


def clear_status() -> None:
    """Mark pipeline as idle (no active job)."""
    write_status(file=None, phase=None, status="idle", error=None)


def process_audio_file(path: Path) -> Dict[str, Optional[Path]]:
    """Run the full pipeline on a single audio file.

    Parameters
    ----------
    path : Path
        Path to an audio file.  Must exist and be non-empty.

    Returns
    -------
    dict
        Keys: ``transcript_path``, ``report_path``, ``archive_path``, ``error``.
        On success ``error`` is ``None``; on failure it holds the exception message
        and the other keys are ``None``.
    """
    if not path.exists():
        return {"transcript_path": None, "report_path": None, "archive_path": None,
                "error": f"File not found: {path}"}

    size = path.stat().st_size
    if size == 0:
        return {"transcript_path": None, "report_path": None, "archive_path": None,
                "error": f"Empty file: {path}"}

    cfg = get_config()
    _temp_dir: Optional[Path] = None

    try:
        file_name = path.name
        write_status(file=file_name, phase="starting", status="processing")
        print(f"[+] Processing {file_name} ({size / (1024*1024):.1f} MB)")

        # ── 1. convert to wav ──────────────────────────────────────
        write_status(file=file_name, phase="converting", status="processing")
        _temp_dir = Path(tempfile.mkdtemp(prefix="audio_proc_"))
        wav_path = to_wav(path, output_dir=_temp_dir)
        print(f"    → Converted to WAV ({wav_path.name})")

        # ── 2. transcribe ──────────────────────────────────────────
        write_status(file=file_name, phase="transcribing", status="processing")
        print(f"    → Starting transcription ({cfg.whisper_model})...", end="", flush=True)
        whisper_result = transcribe(wav_path, model_name=cfg.whisper_model)
        print(" done.")

        # ── 3. diarize ─────────────────────────────────────────────
        write_status(file=file_name, phase="diarizing", status="processing")
        print(f"    → Running speaker diarization...", end="", flush=True)
        turns = diarize(wav_path)
        print(f" done. ({len(turns)} turns)")

        # ── 4. align speakers to words ──────────────────────────────
        write_status(file=file_name, phase="aligning", status="processing")
        print(f"    → Aligning speakers...", end="", flush=True)
        words = words_from_whisper(whisper_result)
        words = assign_speakers(words, turns)
        utterances = aggregate_utterances(words)
        print(f" done. ({len(utterances)} utterances)")

        # ── 5. save transcript ──────────────────────────────────────
        write_status(file=file_name, phase="saving transcript", status="processing")
        cfg.ensure_dirs()
        date_prefix = date.today().strftime("%Y%m%d")
        transcript_path = cfg.transcripts_dir / f"{date_prefix}-{path.stem}.md"

        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(f"# Transcript: {path.stem}\n\n")
            f.write(f"**Source:** {path.name}\n")
            f.write(f"**Processed:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")
            for u in utterances:
                f.write(f"**{u['speaker']}:** {u['text']}\n\n")
        print(f"    → Transcript saved: {transcript_path.name}")

        # ── 6. LLM analysis ────────────────────────────────────────
        write_status(file=file_name, phase="analyzing", status="processing")
        print(f"    → Running LLM analysis...", end="", flush=True)
        report_path = analyze_with_llm(utterances, path.name)
        print(" done.")
        print(f"    → Report saved: {report_path.name}")

        # ── 7. archive original ────────────────────────────────────
        write_status(file=file_name, phase="archiving", status="processing")
        archive_dest = cfg.archive_dir / path.name
        shutil.move(str(path), str(archive_dest))
        print(f"    → Original archived: {archive_dest.name}")

        write_status(file=file_name, phase="completed", status="idle")
        print(f"[✓] Completed {path.name}")
        return {
            "transcript_path": transcript_path,
            "report_path": report_path,
            "archive_path": archive_dest,
            "error": None,
        }

    except Exception as e:
        write_status(file=file_name, phase="error", status="error", error=str(e))
        print(f"[!] Error processing {path.name}: {e}")
        return {
            "transcript_path": None,
            "report_path": None,
            "archive_path": None,
            "error": str(e),
        }

    finally:
        # Clean up temp wav directory
        if _temp_dir is not None and _temp_dir.exists():
            shutil.rmtree(_temp_dir, ignore_errors=True)


def process_multiple(paths: List[Path]) -> List[Dict]:
    """Process several audio files sequentially.

    Each file's result dict includes ``path`` (the original path).
    """
    results = []
    for p in paths:
        result = process_audio_file(p)
        result["path"] = p
        results.append(result)
    return results
