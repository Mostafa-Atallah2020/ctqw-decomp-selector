"""A single progress reporter, so every script's terminal output looks the same.


    from progress import Progress

    prog = Progress(total=len(items), unit="graph")
    for item in items:
        ...
        prog.step(f"n={item.n} {item.dataset}")     # prints one line
    prog.done()                                     # prints the total elapsed

Output:

    [ 45/240] balanced/xgboost seed=2   | 1.4s | elapsed 01m 03s | ETA 04m 22s
    done in 5m 25s

The ETA is a running mean over the steps taken so far, which is right when steps
cost about the same. When they do not, pass a `key` to step() and the estimate is kept per key, so
the cheap early steps do not make the expensive ones look imminent.
"""
from __future__ import annotations

import sys
import time


def fmt_duration(seconds: float) -> str:
    """`1h 04m 17s` / `05m 25s` / `3.2s`: the longest unit that is non-zero."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes:02d}m {secs:02d}s"


class Progress:
    """Per-step progress with elapsed time and an ETA.

    total: how many steps are expected. If it is unknown, pass None and only the
           elapsed time is shown (no ETA, since there is nothing to project).
    unit:  what one step is, for the closing line ("graph", "model", "plot").
    """

    def __init__(self, total: int | None = None, unit: str = "step",
                 stream=sys.stdout):
        self.total = total
        self.unit = unit
        self.stream = stream
        self.done_count = 0
        self.started = time.perf_counter()
        # Per-key timing, so a mix of cheap and expensive steps still gives a
        # sane ETA (see the module docstring).
        self._time_by_key: dict[object, float] = {}
        self._count_by_key: dict[object, int] = {}
        self._remaining_by_key: dict[object, int] = {}
        self._last = self.started

    def plan(self, remaining_by_key: dict) -> None:
        """Declare how many steps of each key are coming, for a per-key ETA."""
        self._remaining_by_key = dict(remaining_by_key)
        if self.total is None:
            self.total = sum(remaining_by_key.values())

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    def _eta(self) -> float | None:
        if not self.done_count or self.total is None:
            return None

        if self._remaining_by_key:
            # ETA per key from its own measured mean, falling back to the slowest
            # measured key for groups not yet seen.
            eta = 0.0
            measured = [self._time_by_key[k] / self._count_by_key[k]
                        for k in self._count_by_key]
            for key, left in self._remaining_by_key.items():
                if left <= 0:
                    continue
                if self._count_by_key.get(key):
                    mean = self._time_by_key[key] / self._count_by_key[key]
                else:
                    mean = max(measured, default=0.0)
                eta += left * mean
            return eta

        mean = self.elapsed / self.done_count
        return mean * (self.total - self.done_count)

    def step(self, label: str = "", key=None) -> None:
        """Record one completed step and print a line for it."""
        now = time.perf_counter()
        took = now - self._last
        self._last = now

        self.done_count += 1
        if key is not None:
            self._time_by_key[key] = self._time_by_key.get(key, 0.0) + took
            self._count_by_key[key] = self._count_by_key.get(key, 0) + 1
            if key in self._remaining_by_key:
                self._remaining_by_key[key] -= 1

        counter = (f"[{self.done_count:>3}/{self.total}]" if self.total
                   else f"[{self.done_count:>3}]")
        eta = self._eta()
        eta_part = f" | ETA {fmt_duration(eta)}" if eta is not None else ""
        print(f"  {counter} {label:<38} | {took:5.1f}s"
              f" | elapsed {fmt_duration(self.elapsed)}{eta_part}",
              file=self.stream, flush=True)

    def done(self) -> None:
        """Print the closing line."""
        n = self.done_count
        unit = self.unit + ("" if n == 1 else "s")
        print(f"  done: {n} {unit} in {fmt_duration(self.elapsed)}",
              file=self.stream, flush=True)
