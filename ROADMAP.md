# Roadmap

Ordered from quickest to build to most involved. Based on feature gaps vs.
commercial tools like the Plaud Note and Soundcore Work.

## 🟢 Trivial (minutes)

- [ ] **Speaker naming** — Map `SPEAKER_00` → real names via config.yaml
- [X] **Auto language detection** — Pass `language=None` to Whisper (it auto-detects)
- [ ] **Meeting type templates** — Selectable prompt templates (meeting, interview, lecture, etc.)
- [ ] **Search across transcripts** — Simple grep alias or Python script over `~/meeting_reports/`

## 🟡 Easy (an hour)

- [ ] **Export (PDF / DOCX)** — Wrapper script around `pandoc` or Python with `pdfkit`
- [ ] **Audio noise reduction** — Integrate `noisereduce` Python package before transcription

## 🟠 Moderate (afternoon)

- [ ] **Custom vocabulary / hotwords** — Pass domain terms to faster-whisper for better accuracy
- [X] **Push notification on completion** — Telegram (or other) bot callback from the pipeline

## 🔴 Bigger build (days)

- [ ] **Web dashboard** — Flask/FastAPI serving reports with search, viewer, mobile-friendly
- [ ] **Real-time / live transcription** — Streaming audio pipeline instead of file-based
- [ ] **Mobile app** — React Native or Flutter app with full viewer, search, notifications
