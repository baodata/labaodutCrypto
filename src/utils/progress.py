"""Throttled progress for long training phases."""

import time


class PhaseProgress:
    def __init__(self, label: str, interval: float = 10.0):
        self.label = label
        self.interval = interval
        self.started = time.perf_counter()
        self.last_print = self.started
        print(f"  {label}: bắt đầu...", flush=True)

    def update(self, completed: int, total: int, detail: str = "", force: bool = False):
        now = time.perf_counter()
        if force or completed == total or now - self.last_print >= self.interval:
            elapsed = now - self.started
            suffix = f" | {detail}" if detail else ""
            print(f"  {self.label}: {completed}/{total} | {elapsed:.1f}s{suffix}", flush=True)
            self.last_print = now
