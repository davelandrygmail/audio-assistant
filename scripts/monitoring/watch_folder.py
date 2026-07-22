"""
watch_folder.py — Watches a directory for new audio files and processes them.

Uses ``watchdog`` to monitor ``~/Recordings`` (configurable in config.yaml).
Files are debounced (2s idle default) then queued for sequential processing
via :mod:`scripts.orchestrator`.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import time
import threading
from pathlib import Path
from queue import Queue, Empty
from typing import Set

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from scripts.utils.config import get_config
from scripts.orchestrator import process_audio_file


class AudioHandler(FileSystemEventHandler):
    def __init__(self):
        self._processing_dir = None  # orchestrator manages its own temp dirs
        self._pending_timers = {}
        self._processing_queue = Queue()
        self._currently_processing: Set[str] = set()
        self._lock = threading.Lock()
        self._cfg = get_config()

        # Start worker thread for sequential processing
        self._worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self._worker_thread.start()

    def _process_queue(self):
        """Worker thread that processes files one at a time."""
        while True:
            try:
                path = self._processing_queue.get(timeout=1)
                if path.exists() and str(path) not in self._currently_processing:
                    self._currently_processing.add(str(path))
                    try:
                        process_audio_file(path)
                    finally:
                        self._currently_processing.remove(str(path))
                self._processing_queue.task_done()
            except Empty:
                continue
            except Exception as e:
                print(f"[!] Queue worker error: {e}")
                continue

    def on_created(self, event):
        if event.is_directory:
            return
        self._schedule(Path(event.src_path))

    def on_modified(self, event):
        if event.is_directory:
            return
        self._schedule(Path(event.src_path))

    def on_moved(self, event):
        if event.is_directory:
            return
        self._schedule(Path(event.dest_path))

    def _schedule(self, path: Path):
        """Debounce: reset a timer on each event so we only process after
        the file has been idle for *debounce_seconds*."""
        if path.suffix.lower() not in self._cfg.supported_extensions:
            return

        with self._lock:
            # Cancel any existing timer for this file
            if path in self._pending_timers:
                self._pending_timers[path].cancel()

            # Start a new timer
            timer = threading.Timer(self._cfg.debounce_seconds, self._enqueue, args=[path])
            self._pending_timers[path] = timer

        timer.start()

    def _enqueue(self, path: Path):
        """Add file to processing queue after debounce."""
        with self._lock:
            if path in self._pending_timers:
                del self._pending_timers[path]

        # Only enqueue if not already being processed
        if str(path) not in self._currently_processing:
            self._processing_queue.put(path)
            print(f"[+] Queued {path.name} for processing")


def start_watcher():
    cfg = get_config()
    cfg.watch_dir.mkdir(parents=True, exist_ok=True)

    handler = AudioHandler()
    observer = Observer()
    observer.schedule(handler, str(cfg.watch_dir), recursive=True)
    observer.start()

    print(f"👀 Watching {cfg.watch_dir} for new audio files...")
    print(f"   Supported formats: {', '.join(sorted(cfg.supported_extensions))}")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n⏹️  Stopping watcher...")
        observer.stop()
    observer.join()

    print("✓ Watcher stopped")


if __name__ == "__main__":
    start_watcher()
