# scripts/processing/transcribe.py
import sys
from pathlib import Path
from typing import List, Dict

from faster_whisper import WhisperModel
from typing import Literal

# ------------------------------------------------------------
#  Helper to load the model only once (lazy‑load pattern)
# ------------------------------------------------------------
# Global singleton – the model object lives for the whole process.
_loaded_model = None
_model_name   = None   # cache the name we loaded

def _load_whisper_model(name: Literal[
    "tiny", "base", "small", "medium", "large", "large-v2",
    "distil-large-v2", "distil-large-v3", "distil-medium.en",
    "distil-small.en"
] = "distil-large-v2") -> WhisperModel:
    """
    Load a Faster-Whisper / Distil-Whisper model.

    Distil-Whisper models are ~6x faster than standard Whisper with ~95% accuracy.
    """
    global _loaded_model, _model_name
    if _loaded_model is None:
        # Use CPU with INT8 quantization for best speed/accuracy tradeoff
        _loaded_model = WhisperModel(
            name,
            device="cpu",
            compute_type="int8"
        )
        _model_name = name
        print(f"[transcribe] Faster-Whisper model '{name}' loaded (CPU, int8).")
    return _loaded_model

# ------------------------------------------------------------
#  Public API – transcribe a wav file (or any audioread file)
# ------------------------------------------------------------
def transcribe(
    wav_path: Path,
    *,
    model_name: Literal[
        "tiny", "base", "small", "medium", "large", "large-v2",
        "distil-large-v2", "distil-large-v3", "distil-medium.en",
        "distil-small.en"
    ] = "distil-large-v2"
) -> Dict:
    """
    Transcribe an audio file that has already been converted to PCM-WAV.

    Returns a dictionary matching the OpenAI Whisper output format for compatibility:
    {
        "text": "full transcript",
        "segments": [{"start", "end", "text", "words": [{"word", "start", "end"}]}],
        "words": [{"word", "start", "end"}]  # flattened list
    }
    """
    model = _load_whisper_model(model_name)

    # Faster-Whisper transcribe with progress indicator
    print(f"    → Starting transcription (distil-large-v2)...", end="", flush=True)

    segments_iter, info = model.transcribe(
        str(wav_path),
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    # Convert to OpenAI Whisper format for compatibility with downstream code
    segments = []
    all_words = []
    full_text_parts = []

    for segment in segments_iter:
        seg_words = []
        if hasattr(segment, 'words') and segment.words:
            for word in segment.words:
                seg_words.append({
                    "word": word.word,
                    "start": word.start,
                    "end": word.end
                })
                all_words.append({
                    "word": word.word,
                    "start": word.start,
                    "end": word.end
                })

        segments.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text,
            "words": seg_words
        })
        full_text_parts.append(segment.text)

    print(f" done.")

    result = {
        "text": " ".join(full_text_parts),
        "segments": segments,
        "words": all_words
    }

    return result
