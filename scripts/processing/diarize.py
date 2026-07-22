# file: diarize.py
import os
from pathlib import Path
from pyannote.audio import Pipeline
import torch

from scripts.utils.config import get_config

# Cache the loaded pipeline so we don't reload weights every call
_loaded_pipeline = None

def get_diarization_pipeline():
    global _loaded_pipeline
    if _loaded_pipeline is None:
        cfg = get_config()
        token = cfg.hf_token
        if not token:
            raise ValueError(
                "HF_TOKEN is not set. "
                "Add it to .env (see .env.example) or export it in your shell."
            )
        _loaded_pipeline = Pipeline.from_pretrained(
            cfg.diarization_model,
            token=token,
        )
    return _loaded_pipeline

def diarize(wav_path: Path):
    pipeline = get_diarization_pipeline()
    # pyannote v4 returns a DiarizeOutput dataclass with .speaker_diarization
    output = pipeline(str(wav_path))
    
    if hasattr(output, 'speaker_diarization'):
        # pyannote v4: DiarizeOutput has the Annotation directly
        diarization = output.speaker_diarization
    elif hasattr(output, 'itertracks'):
        # pyannote v3 or legacy: output is already an Annotation
        diarization = output
    else:
        raise ValueError(f"Unexpected pyannote output type: {type(output)}")
    
    # Convert annotation to list of turns
    turns = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turns.append({"start": turn.start, "end": turn.end, "speaker": speaker})
    return turns