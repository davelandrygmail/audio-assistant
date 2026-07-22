# Audio Assistant 🎙️

An automated audio processing pipeline that watches a folder for recordings, transcribes them with speaker attribution, and generates structured AI meeting reports — all hands-free.

Drop an audio file in `~/Recordings/` and the pipeline handles the rest: conversion, transcription, speaker diarization, LLM analysis, and archiving.

## Pipeline

```
🎧 Audio file dropped
  ↓
🔄 ffmpeg → 16 kHz mono WAV
  ↓
📝 faster-whisper (distil-large-v2, CPU int8)
  ↓
🗣️ pyannote speaker diarization
  ↓
🔗 Word-level speaker alignment
  ↓
📄 Speaker-labeled transcript saved
  ↓
🧠 LLM analysis (OpenRouter free-tier)
  ↓
📋 Structured report saved
  └── Original file archived
```

## Project Structure

```
audio-assistant/
├── scripts/
│   ├── orchestrator.py          # Central pipeline entrypoint
│   ├── processing/
│   │   ├── convert_wav.py       # ffmpeg → 16 kHz mono WAV
│   │   ├── transcribe.py        # faster-whisper transcription
│   │   └── diarize.py           # pyannote speaker diarization
│   ├── ai/
│   │   └── llm_analysis.py      # OpenRouter LLM report generation
│   ├── utils/
│   │   ├── align.py             # Speaker-to-word alignment
│   │   └── config.py            # Config loader (config.yaml + .env)
│   └── monitoring/
│       └── watch_folder.py      # Folder watcher daemon
├── config.yaml                  # Non-secret settings
├── .env.example                 # Secret key template
└── requirements.txt
```

## Prerequisites

- **Python 3.10+**
- **ffmpeg** (for audio conversion)
- **Hugging Face token** — [hf.co/settings/tokens](https://huggingface.co/settings/tokens). Required because `pyannote/speaker-diarization-3.1` is a gated model; you must accept its terms on Hugging Face.
- **OpenRouter API key** — [openrouter.ai/keys](https://openrouter.ai/keys). Used for LLM analysis (free-tier available).

## Setup

```bash
# 1. Clone the repo
git clone git@github.com:davelandrygmail/audio-assistant.git
cd audio-assistant

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up your secrets
cp .env.example .env
# Edit .env with your real HF_TOKEN and OPENROUTER_API_KEY

# 5. Verify (optional)
python -c "
import sys; sys.path.insert(0, '.')
from scripts.utils.config import get_config
cfg = get_config()
print(f'HF_TOKEN set:      {bool(cfg.hf_token)}')
print(f'OpenRouter set:    {bool(cfg.openrouter_api_key)}')
print(f'Watching:          {cfg.watch_dir}')
"
```

## Configuration

### `config.yaml` — non-secret settings

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `paths.watch_dir` | `~/Recordings` | Directory to watch for audio files |
| `paths.transcripts_dir` | `~/meeting_reports/transcripts` | Speaker-labeled transcript output |
| `paths.reports_dir` | `~/meeting_reports` | LLM analysis report output |
| `paths.archive_dir` | `~/meeting_reports/original_recording` | Where processed originals are moved |
| `models.whisper` | `distil-large-v2` | Whisper model variant |
| `models.whisper_device` | `cpu` | Device for inference |
| `models.whisper_compute_type` | `int8` | INT8 quantization for speed |
| `llm.model` | `openrouter/free` | OpenRouter model |
| `llm.temperature` | `0.2` | LLM generation temperature |
| `processing.debounce_seconds` | `2.0` | Wait time before processing a new file |
| `processing.supported_extensions` | `[.mp3, .wav, .m4a, ...]` | Accepted audio formats |

### `.env` — secrets (never committed)

```
HF_TOKEN=hf_your_token_here
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

## Usage

### Daemon mode (watcher)

```bash
source venv/bin/activate
python scripts/monitoring/watch_folder.py
```

Drop audio files into `~/Recordings/` — the watcher picks them up, processes them sequentially, and moves originals to the archive when done.

### One-shot mode

```bash
source venv/bin/activate
python -c "
import sys; sys.path.insert(0, '.')
from scripts.orchestrator import process_audio_file
result = process_audio_file('path/to/audio.mp3')
print(result)
"
```

## Output

```
~/meeting_reports/
├── 20260722-meeting_name.md              # LLM analysis report
├── transcripts/
│   └── 20260722-meeting_name.md          # Speaker-labeled transcript
└── original_recording/
    └── meeting_name.mp3                  # Archived original
```

LLM reports include:
- Executive Summary
- Decisions Made
- Risks & Open Questions
- Action Items & Deadlines
- People & Technologies Mentioned
- Tags

## Running as a Systemd Service

```ini
# ~/.config/systemd/user/audio-assistant.service
[Unit]
Description=Audio Assistant Watcher
After=network-online.target

[Service]
Type=simple
ExecStart=%h/venv/bin/python %h/audio-assistant/scripts/monitoring/watch_folder.py
WorkingDirectory=%h/audio-assistant
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
```

```bash
systemctl --user daemon-reload
systemctl --user enable --now audio-assistant
journalctl --user -u audio-assistant -f
```

Enable lingering so the service survives logout:

```bash
sudo loginctl enable-linger $USER
```

## Tech Stack

| Component | Tool |
|-----------|------|
| Transcription | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (distil-large-v2) |
| Diarization | [pyannote.audio](https://github.com/pyannote/pyannote-audio) 3.1 |
| LLM | OpenRouter (free-tier, via OpenAI SDK) |
| Audio conversion | ffmpeg |
| File watching | watchdog |
