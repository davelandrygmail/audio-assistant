# file: align.py
from typing import List, Dict

def words_from_whisper(whisper_result: dict) -> List[Dict]:
    """Flatten Whisper's word‑level timestamps."""
    words = []
    for seg in whisper_result["segments"]:
        for w in seg["words"]:
            words.append({"word": w["word"], "start": w["start"], "end": w["end"]})
    return words

def assign_speakers(words: List[Dict], turns: List[Dict]) -> List[Dict]:
    """For each word, pick the speaker whose turn covers its midpoint."""
    for w in words:
        mid = (w["start"] + w["end"]) / 2.0
        # find first turn that contains mid
        speaker = "UNKNOWN"
        for t in turns:
            if t["start"] <= mid < t["end"]:
                speaker = t["speaker"]
                break
        w["speaker"] = speaker
    return words

def aggregate_utterances(words: List[Dict]) -> List[Dict]:
    """Group consecutive words with same speaker into utterances."""
    utterances = []
    current = None
    for w in words:
        if current is None or w["speaker"] != current["speaker"]:
            # start a new utterance
            current = {
                "speaker": w["speaker"],
                "start": w["start"],
                "end": w["end"],
                "text": w["word"].strip()
            }
            utterances.append(current)
        else:
            # extend current utterance
            current["end"] = w["end"]
            current["text"] += " " + w["word"].strip()
    # clean up spaces
    for u in utterances:
        u["text"] = u["text"].strip()
    return utterances