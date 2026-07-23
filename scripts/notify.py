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
from pathlib import Path
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


def notify_phase(file: str, phase: str, status: str,
                error: Optional[str] = None,
                attach_path: Optional[Path] = None) -> None:
    """Send a ntfy.sh notification for a pipeline phase transition.

    If *attach_path* is provided (e.g. the report file on completion)
    it is uploaded as a file attachment that subscribers can download.
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

    _send(topic, message, attach_path=attach_path)


def _send(topic: str, message: str, attach_path: Optional[Path] = None) -> None:
    """POST a notification to ntfy.sh, optionally with a file attachment.

    Attachments are sent as the raw request body with ``Message``,
    ``Filename`` and ``Content-Type`` headers (ntfy's expected format).
    Falls back to text-only if the attachment can't be uploaded.
    """
    import urllib.request
    import urllib.error

    url = f"https://ntfy.sh/{topic}"

    if attach_path and attach_path.exists() and attach_path.stat().st_size > 0:
        try:
            import requests
            # HTTP headers must be ASCII — strip emoji / non-ASCII for the header
            ascii_message = message.encode("ascii", errors="replace").decode("ascii")
            with open(attach_path, "rb") as f:
                resp = requests.post(
                    url,
                    data=f,
                    headers={
                        "Message": ascii_message,
                        "Filename": attach_path.name,
                        "Content-Type": "text/markdown",
                    },
                    timeout=30,
                )
            if not resp.ok:
                log.warning("ntfy attachment upload failed: HTTP %s", resp.status_code)
            return
        except ImportError:
            log.warning("requests not available — falling back to text-only")
        except Exception as e:
            log.warning("ntfy attachment upload error: %s", e)
            # fall through to text-only

    # Text-only fallback
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
