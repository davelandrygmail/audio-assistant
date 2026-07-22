"""
Configuration loader for the Audio Assistant.

Loads settings from two sources:
  1. config.yaml   — non-secret settings (paths, model names, etc.)
  2. .env          — secrets (API keys, tokens)

.env is loaded first, then its values are exposed as os.environ entries
so existing code that reads os.environ works unchanged.

Usage:
    from scripts.utils.config import get_config

    cfg = get_config()
    cfg.hf_token           # str or None
    cfg.openrouter_api_key # str or None
    cfg.watch_dir          # Path
    cfg.whisper_model      # str
    ...
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, Any, Optional
import yaml


# ── paths ──────────────────────────────────────────
_PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
_CONFIG_YAML  = _PROJECT_DIR / "config.yaml"
_DOTENV       = _PROJECT_DIR / ".env"


# ── .env parser (no external dependency) ───────────
def _load_dotenv(path: Path) -> Dict[str, str]:
    """Parse a simple KEY=VALUE file (no interpolation, no quotes)."""
    if not path.exists():
        return {}
    env = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = re.match(r"^([A-Za-z_]\w*)=(.*)$", line)
            if m:
                env[m.group(1)] = m.group(2).strip().strip("\"'")
    return env


# ── config object ──────────────────────────────────
class _Config:
    """Singleton-holder for the loaded config.

    Access via :func:`get_config` — never instantiate directly.
    """

    _instance: Optional["_Config"] = None

    def __init__(self):
        dotenv_vals = _load_dotenv(_DOTENV)

        # Seed os.environ with .env values (won't override existing)
        for k, v in dotenv_vals.items():
            os.environ.setdefault(k, v)

        # Load YAML
        if _CONFIG_YAML.exists():
            with open(_CONFIG_YAML, encoding="utf-8") as f:
                raw: Dict[str, Any] = yaml.safe_load(f) or {}
        else:
            raw = {}

        # ── secrets ──
        self.hf_token: Optional[str] = os.environ.get("HF_TOKEN")
        self.openrouter_api_key: Optional[str] = os.environ.get("OPENROUTER_API_KEY")

        # ── paths (expand ~ and resolve) ──
        paths = raw.get("paths", {})
        self.watch_dir: Path = Path(paths.get("watch_dir", "~/Recordings")).expanduser()
        self.transcripts_dir: Path = Path(paths.get("transcripts_dir", "~/meeting_reports/transcripts")).expanduser()
        self.reports_dir: Path = Path(paths.get("reports_dir", "~/meeting_reports")).expanduser()
        self.archive_dir: Path = Path(paths.get("archive_dir", "~/meeting_reports/original_recording")).expanduser()

        # ── models ──
        models = raw.get("models", {})
        self.whisper_model: str = models.get("whisper", "distil-large-v2")
        self.whisper_device: str = models.get("whisper_device", "cpu")
        self.whisper_compute_type: str = models.get("whisper_compute_type", "int8")
        self.diarization_model: str = models.get("diarization", "pyannote/speaker-diarization-3.1")

        # ── LLM ──
        llm = raw.get("llm", {})
        self.llm_provider: str = llm.get("provider", "openrouter")
        self.llm_base_url: str = llm.get("base_url", "https://openrouter.ai/api/v1")
        self.llm_model: str = llm.get("model", "openrouter/free")
        self.llm_temperature: float = llm.get("temperature", 0.2)

        # ── processing ──
        proc = raw.get("processing", {})
        self.debounce_seconds: float = proc.get("debounce_seconds", 2.0)
        self.supported_extensions: set[str] = set(
            proc.get("supported_extensions", [".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm"])
        )

    # ── helpers ──
    def ensure_dirs(self):
        """Create all output directories if they don't exist."""
        for d in (self.transcripts_dir, self.reports_dir, self.archive_dir):
            d.mkdir(parents=True, exist_ok=True)


_CONFIG_CACHE: Optional[_Config] = None


def get_config() -> _Config:
    """Return the singleton config object."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        _CONFIG_CACHE = _Config()
    return _CONFIG_CACHE
