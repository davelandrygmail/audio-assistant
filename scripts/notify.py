"""
notify.py — Push notifications via ntfy.sh.

Sends HTTP POST requests to https://ntfy.sh/{topic} for pipeline events.
No additional dependencies required — uses Python stdlib ``urllib``.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from typing import Optional

from scripts.utils.config import get_config

log = logging.getLogger(__name__)

EMOJI_MAP = {
    "starting":     "🎬",
    "converting":   "🔄",
    "transcribing": "📝",
    "diarizing":    "🗣️",
    "aligning":     "🔗",
    "saving transcript": "💾",
    "analyzing":    "🧠",
    "archiving":    "📦",
    "completed":    "✅",
    "error":        "❌",
    "idle":         "💤",
}


def notify_phase(file: str, phase: str, status: str, error: Optional[str] = None) -> None:
    """Send a ntfy.sh notification for a pipeline phase transition.

    Silently does nothing if ``notifications.ntfy_topic`` is not set in config.
    """
    cfg = get_config()
    topic = cfg.ntfy_topic
    if not topic:
        return

    emoji = EMOJI_MAP.get(phase if status == "processing" else status, "ℹ️")

    if status == "error":
        message = f"{emoji} Error on **{file}**: {error}"
    elif status == "idle" and phase == "completed":
        message = f"{emoji} Finished **{file}** — transcript + report ready"
    elif status == "processing" and phase == "starting":
        message = f"{emoji} Processing **{file}**"
    elif status == "processing":
        message = f"{emoji} {phase.capitalize()} **{file}**"
    else:
        return  # don't notify on idle clear, etc.

    _send(topic, message)


def _send(topic: str, message: str) -> None:
    """POST the message to ntfy.sh via urllib (no external deps)."""
    import urllib.request
    import urllib.error

    url = f"https://ntfy.sh/{topic}"
    data = message.encode("utf-8")

    try:
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "text/plain; charset=utf-8"},
        )
        urllib.request.urlopen(req, timeout=10)
    except urllib.error.URLError as e:
        log.warning("ntfy notification failed: %s", e)
    except Exception as e:
        log.warning("ntfy notification error: %s", e)
