#!/usr/bin/env python3
"""
status.py — Check the current pipeline status.

Usage:
    python scripts/status.py          # pretty-print
    python scripts/status.py --json   # raw JSON (for scripting)
"""

import json
import sys
from pathlib import Path

STATUS_FILE = Path(__file__).resolve().parent.parent / "status.json"
EMOJI = {
    "idle":         "💤",
    "processing":   "⚙️",
    "error":        "❌",
    "starting":     "🎬",
    "converting":   "🔄",
    "transcribing": "📝",
    "diarizing":    "🗣️",
    "aligning":     "🔗",
    "saving transcript": "💾",
    "analyzing":    "🧠",
    "archiving":    "📦",
    "completed":    "✅",
}


def pretty(data: dict) -> str:
    status = data.get("status", "unknown")
    phase = data.get("phase") or status
    emoji = EMOJI.get(phase, EMOJI.get(status, "❓"))

    lines = [f"{emoji}  Pipeline: {status.upper()}"]
    if data.get("file"):
        lines.append(f"   File:  {data['file']}")
    if data.get("phase") and status != "idle":
        lines.append(f"   Phase: {data['phase']}")
    if data.get("pid"):
        lines.append(f"   PID:   {data['pid']}")
    if data.get("timestamp"):
        from datetime import datetime
        ts = datetime.fromisoformat(data["timestamp"])
        local = ts.astimezone().strftime("%H:%M:%S")
        lines.append(f"   Since: {local}")
    if data.get("error"):
        lines.append(f"   Error: {data['error']}")
    return "\n".join(lines)


def main():
    if not STATUS_FILE.exists():
        print("💤  No status file — pipeline has never run.")
        sys.exit(0)

    raw = STATUS_FILE.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        print("⚠️  Status file corrupted.")
        sys.exit(1)

    if "--json" in sys.argv:
        print(raw.strip())
    else:
        print(pretty(data))


if __name__ == "__main__":
    main()
