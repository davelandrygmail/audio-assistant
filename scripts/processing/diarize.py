"""
diarize.py — Speaker diarization with two configurable backends.

Method is toggled via config.yaml:

  ``"lightweight"`` (default) — Silero VAD + speechbrain ECAPA embeddings + sklearn clustering.
  ``"pyannote"``               — Original full pyannote pipeline (slower, fallback).

The function signature and return format are identical regardless of backend:
    diarize(wav_path: Path) -> List[Dict[str, float | str]]
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torchaudio

from scripts.utils.config import get_config

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  Lightweight backend  (default)
# ═══════════════════════════════════════════════════════════════════════

_VAD = None          # Silero VAD singleton
_ECAPA = None        # SpeechBrain ECAPA singleton


def _load_vad():
    global _VAD
    if _VAD is None:
        import silero_vad
        _VAD = silero_vad.load_silero_vad()
    return _VAD


def _load_ecapa():
    global _ECAPA
    if _ECAPA is None:
        from speechbrain.inference.speaker import SpeakerRecognition
        _ECAPA = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
        )
    return _ECAPA


def _resample_if_needed(wav: torch.Tensor, sr: int) -> tuple[torch.Tensor, int]:
    """Downmix to mono and resample to 16 kHz if necessary."""
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        wav = resampler(wav)
        sr = 16000
    return wav, sr


def _speech_segments(wav_path: Path) -> List[Dict[str, float]]:
    """Return speech / non-speech boundaries using Silero VAD.

    Each dict has ``start`` and ``end`` in seconds.
    Segments shorter than 300 ms are discarded.
    """
    model = _load_vad()
    wav, sr = torchaudio.load(str(wav_path))
    wav, sr = _resample_if_needed(wav, sr)

    # Silero expects a 1-D signal
    speech_ts = model.get_speech_timestamps(wav[0], sampling_rate=sr)

    segments = []
    for ts in speech_ts:
        start_s = ts["start"] / sr
        end_s = ts["end"] / sr
        if end_s - start_s >= 0.3:
            segments.append({"start": start_s, "end": end_s})

    return segments


def _extract_embeddings(wav_path: Path, segments: List[Dict]) -> np.ndarray:
    """Extract one speaker embedding (ECAPA 192-d) per speech segment.

    Returns an array of shape ``(N, 192)`` or an empty array if no valid
    segments remain.
    """
    classifier = _load_ecapa()
    wav, fs = torchaudio.load(str(wav_path))
    wav, fs = _resample_if_needed(wav, fs)

    embeddings: List[np.ndarray] = []
    for seg in segments:
        start_s = int(seg["start"] * fs)
        end_s = int(seg["end"] * fs)
        chunk = wav[:, start_s:end_s]

        # Skip chunks shorter than 500 ms — too little voice data
        if chunk.shape[1] < int(0.5 * fs):
            continue

        with torch.no_grad():
            emb = classifier.encode_batch(chunk)  # (1, 1, emb_dim)
        embeddings.append(emb.squeeze().cpu().numpy())

    return np.stack(embeddings) if embeddings else np.array([])


def _cluster_embeddings(embeddings: np.ndarray) -> List[str]:
    """Assign speaker labels via agglomerative clustering."""
    if embeddings.shape[0] <= 1:
        return [f"SPEAKER_{i:02d}" for i in range(embeddings.shape[0])]

    from sklearn.cluster import AgglomerativeClustering

    cfg = get_config()
    threshold = cfg.diarization_clustering_threshold

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
    ).fit(embeddings)

    return [f"SPEAKER_{l:02d}" for l in clustering.labels_]


def diarize_lightweight(wav_path: Path) -> List[Dict[str, float | str]]:
    """Run the full lightweight diarization pipeline.

    VAD → speaker embeddings → agglomerative clustering → turns.
    """
    segments = _speech_segments(wav_path)
    if not segments:
        log.warning("No speech segments found in %s", wav_path.name)
        return []

    embeddings = _extract_embeddings(wav_path, segments)
    if embeddings.shape[0] < 1:
        log.warning("No valid embeddings extracted from %s", wav_path.name)
        return []

    labels = _cluster_embeddings(embeddings)

    # Zip labels back to the corresponding segments (may have filtered some)
    turns = []
    for seg, label in zip(segments[: len(labels)], labels):
        turns.append({
            "start": seg["start"],
            "end": seg["end"],
            "speaker": label,
        })

    return turns


# ═══════════════════════════════════════════════════════════════════════
#  Pyannote backend  (original, slower fallback)
# ═══════════════════════════════════════════════════════════════════════

_PYANNOTE = None


def _load_pyannote():
    global _PYANNOTE
    if _PYANNOTE is None:
        from pyannote.audio import Pipeline

        cfg = get_config()
        token = cfg.hf_token
        if not token:
            raise ValueError(
                "HF_TOKEN is not set. Add it to .env (see .env.example) "
                "or export it in your shell."
            )
        _PYANNOTE = Pipeline.from_pretrained(cfg.diarization_model, token=token)
    return _PYANNOTE


def diarize_pyannote(wav_path: Path) -> List[Dict[str, float | str]]:
    """Original pyannote-based diarization (slower but established)."""
    pipeline = _load_pyannote()
    output = pipeline(str(wav_path))

    if hasattr(output, "speaker_diarization"):
        diarization = output.speaker_diarization
    else:
        diarization = output

    turns = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turns.append({"start": turn.start, "end": turn.end, "speaker": speaker})

    return turns


# ═══════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════

def diarize(wav_path: Path) -> List[Dict[str, float | str]]:
    """Route to the configured diarization backend.

    Parameters
    ----------
    wav_path : Path
        Path to a 16 kHz mono WAV file.

    Returns
    -------
    list[dict]
        Each dict: ``{"start": float, "end": float, "speaker": str}``

    Raises
    ------
    ValueError
        If ``diarization.method`` in ``config.yaml`` is not recognised.
    """
    cfg = get_config()
    method = cfg.diarization_method

    if method == "lightweight":
        return diarize_lightweight(wav_path)
    elif method == "pyannote":
        return diarize_pyannote(wav_path)
    else:
        raise ValueError(
            f"Unknown diarization method {method!r}. "
            "Use 'lightweight' or 'pyannote' in config.yaml."
        )
