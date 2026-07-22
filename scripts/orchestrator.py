"""
orchestrator.py — Central entrypoint for the audio processing pipeline.

Call :func:`process_audio_file` to run the full chain on a single file:
    convert → transcribe → diarize → align → analyze + save outputs.

This module extracts the pipeline logic from the watchdog so it can be
invoked directly (CLI, one-off, cron, test) or driven by the folder watcher.
"""

from __future__ import annotations

import shutil
import tempfile
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

from scripts.utils.config import get_config
from scripts.processing.convert_wav import to_wav
from scripts.processing.transcribe import transcribe
from scripts.processing.diarize import diarize
from scripts.utils.align import words_from_whisper, assign_speakers, aggregate_utterances
from scripts.ai.llm_analysis import analyze_with_llm


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
        print(f"[+] Processing {path.name} ({size / (1024*1024):.1f} MB)")

        # ── 1. convert to wav ──────────────────────────────────────
        _temp_dir = Path(tempfile.mkdtemp(prefix="audio_proc_"))
        wav_path = to_wav(path, output_dir=_temp_dir)
        print(f"    → Converted to WAV ({wav_path.name})")

        # ── 2. transcribe ──────────────────────────────────────────
        print(f"    → Starting transcription ({cfg.whisper_model})...", end="", flush=True)
        whisper_result = transcribe(wav_path, model_name=cfg.whisper_model)
        print(" done.")

        # ── 3. diarize ─────────────────────────────────────────────
        print(f"    → Running speaker diarization...", end="", flush=True)
        turns = diarize(wav_path)
        print(f" done. ({len(turns)} turns)")

        # ── 4. align speakers to words ──────────────────────────────
        print(f"    → Aligning speakers...", end="", flush=True)
        words = words_from_whisper(whisper_result)
        words = assign_speakers(words, turns)
        utterances = aggregate_utterances(words)
        print(f" done. ({len(utterances)} utterances)")

        # ── 5. save transcript ──────────────────────────────────────
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
        print(f"    → Running LLM analysis...", end="", flush=True)
        report_path = analyze_with_llm(utterances, path.name)
        print(" done.")
        print(f"    → Report saved: {report_path.name}")

        # ── 7. archive original ────────────────────────────────────
        archive_dest = cfg.archive_dir / path.name
        shutil.move(str(path), str(archive_dest))
        print(f"    → Original archived: {archive_dest.name}")

        print(f"[✓] Completed {path.name}")
        return {
            "transcript_path": transcript_path,
            "report_path": report_path,
            "archive_path": archive_dest,
            "error": None,
        }

    except Exception as e:
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
