# scripts/processing/convert_wav.py
import subprocess
import sys
from pathlib import Path
from typing import Tuple

def to_wav(src_path: Path, output_dir: Path = None) -> Path:
    """
    Convert *any* audio format (mp3, m4a, flac, ogg, etc.) to a 16 kHz,
    mono, 16-bit PCM wav file.

    Parameters
    ----------
    src_path: Path
        Path to the original audio file.
    output_dir: Path, optional
        Directory to write the output wav. If None, writes next to source.

    Returns
    -------
    Path
        Absolute path of the newly created .wav file.

    Raises
    ------
    RuntimeError
        If ffmpeg exits with a non-zero status.
    FileNotFoundError
        If ffmpeg executable cannot be found on PATH.
    """
    # Determine output directory
    if output_dir is None:
        output_dir = src_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Destination path
    dst_path = output_dir / (src_path.stem + ".wav")

    # If input IS already the destination path, use a temp file to avoid
    # ffmpeg trying to read and write the same file simultaneously.
    tmp_path = None
    if src_path.resolve() == dst_path.resolve():
        tmp_path = src_path.with_name(src_path.stem + "_converted.wav")
        output_path = tmp_path
    else:
        output_path = dst_path

    # ffmpeg command – -y overwrites an existing file silently.
    cmd = [
        "ffmpeg",
        "-y",                              # overwrite without prompting
        "-i", str(src_path),               # input file
        "-ar", "16000",                    # set sample_rate to 16 kHz
        "-ac", "1",                        # set channels to mono
        "-f", "wav",                       # force output format (wav)
        str(output_path)                   # output file
    ]

    # Run ffmpeg, capture only errors – keep output quiet.
    subprocess.run(
        cmd,
        check=True,                         # raise if ffmpeg returns != 0
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # If we used a temp file, move it over the original.
    if tmp_path is not None:
        tmp_path.replace(dst_path)

    return dst_path

def convert_batch(paths: list[Path]) -> list[Path]:
    """
    Helper that runs `to_wav` on many files and returns the list of
    created wav paths.  Useful when you want to batch‑process a folder
    without triggering the watchdog event loop.
    """
    converted = []
    for p in paths:
        if not p.is_file():
            continue
        try:
            wav = to_wav(p)
            converted.append(wav)
        except Exception as exc:
            print(f"[convert_wav] FAILED: {p.name} – {exc}", file=sys.stderr)
    return converted