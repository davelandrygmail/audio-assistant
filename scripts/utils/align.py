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
    """For each word, pick the speaker whose turn covers its midpoint.

    Words whose midpoint falls outside all diarization turns (gaps
    between VAD segments) are assigned to the **nearest** turn by
    temporal proximity instead of being left as ``UNKNOWN``.
    """
    if not turns:
        for w in words:
            w["speaker"] = "UNKNOWN"
        return words

    # ── primary: exact overlap ────────────────────────────────────
    orphan_indices: List[int] = []
    for i, w in enumerate(words):
        mid = (w["start"] + w["end"]) / 2.0
        speaker = None
        for t in turns:
            if t["start"] <= mid < t["end"]:
                speaker = t["speaker"]
                break
        if speaker is not None:
            w["speaker"] = speaker
        else:
            orphan_indices.append(i)

    if not orphan_indices:
        return words

    # ── fallback: nearest turn for orphan words ───────────────────
    for i in orphan_indices:
        w = words[i]
        mid = (w["start"] + w["end"]) / 2.0
        # Find turn with minimum distance to this word's midpoint
        best_turn = min(turns, key=lambda t: min(
            abs(mid - t["start"]), abs(mid - t["end"])
        ))
        w["speaker"] = best_turn["speaker"]

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