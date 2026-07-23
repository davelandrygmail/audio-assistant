# file: llm_analysis.py
import os
from pathlib import Path
from openai import OpenAI   # pip install openai
from typing import List, Dict

from scripts.utils.config import get_config


def _get_client() -> OpenAI:
    cfg = get_config()
    api_key = cfg.openrouter_api_key
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY is not set. "
            "Add it to .env (see .env.example) or export it in your shell."
        )
    return OpenAI(
        base_url=cfg.llm_base_url,
        api_key=api_key,
    )

def format_transcript_for_prompt(utterances: List[Dict]) -> str:
    """Create a readable block: [Speaker] (mm:ss) text"""
    lines = []
    for u in utterances:
        # mm:ss format
        start_min = int(u["start"] // 60)
        start_sec = int(u["start"] % 60)
        lines.append(f"[{u['speaker']}] ({start_min:02d}:{start_sec:02d}) {u['text']}")
    return "\n".join(lines)

PROMPT_TEMPLATE = """You are an executive assistant.

Analyze this meeting.

Produce:

1. Executive Summary
2. Decisions Made
3. Risks
4. Open Questions
5. Action Items
6. Deadlines
7. People Mentioned
8. Technologies Mentioned
9. One-paragraph summary
10. Tags

Meeting transcript:
{transcript}
"""

def analyze_with_llm(utterances: List[Dict], meeting_name: str) -> Path:
    transcript_block = format_transcript_for_prompt(utterances)
    prompt = PROMPT_TEMPLATE.format(transcript=transcript_block)

    cfg = get_config()
    client = _get_client()
    markdown_report: Optional[str] = None

    MAX_RETRIES = 3
    RETRY_DELAY_S = 15  # seconds between retries

    import time

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=cfg.llm_model,
                messages=[
                    {"role": "system", "content": "You are a precise executive‑assistant that follows the requested output format exactly."},
                    {"role": "user", "content": prompt}
                ],
                temperature=cfg.llm_temperature,
            )
            markdown_report = response.choices[0].message.content

            if markdown_report is not None:
                # Safety/content-moderation responses are typically very short
                # (e.g. "User Safety: safe") — treat them as failures and retry.
                safe = markdown_report.strip().lower()
                if len(safe) < 60 or safe.startswith("user safety"):
                    print(f"    ⚠  LLM returned safety/moderation response "
                          f"(attempt {attempt}/{MAX_RETRIES})")
                    markdown_report = None
                else:
                    break  # success

            # null or safety — retry
            if markdown_report is None:
                print(f"    ⚠  LLM returned no valid content "
                      f"(attempt {attempt}/{MAX_RETRIES})")

        except Exception as e:
            print(f"    ⚠  LLM API error (attempt {attempt}/{MAX_RETRIES}): {e}")

        if attempt < MAX_RETRIES:
            print(f"       retrying in {RETRY_DELAY_S}s...")
            time.sleep(RETRY_DELAY_S)

    # All attempts exhausted — write fallback
    if markdown_report is None:
        markdown_report = (
            "# LLM Analysis Failed\n\n"
            f"The LLM was unreachable after {MAX_RETRIES} attempts.\n\n"
            "The transcript was saved successfully — re-run analysis when the provider is available.\n"
        )
        print(f"    ✗  LLM unavailable after {MAX_RETRIES} attempts — fallback report written")

    out_dir = cfg.reports_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    from datetime import date
    date_prefix = date.today().strftime("%Y%m%d")
    out_path = out_dir / f"{date_prefix}-{Path(meeting_name).stem}.md"
    out_path.write_text(markdown_report, encoding="utf-8")
    print(f"    → Report written to {out_path}")
    return out_path