import time
import os
import atexit
import functools
from contextlib import contextmanager


class Timer:
    def __init__(self, auto_report=False, report_fn=print):
        self.times = []  # Backwards compat: aggregate of all calls
        self._name_to_times = {}
        self._auto_report = auto_report
        self._report_fn = report_fn
        if auto_report:
            atexit.register(self._report_summary)

    def __call__(self, func=None, *, name=None):
        # Supports both @timer and @timer(name="stage")
        if func is None:
            return lambda f: self._wrap(f, name)
        return self._wrap(func, name)

    def _wrap(self, func, name):
        label = name or getattr(func, "__qualname__", getattr(func, "__name__", "<unknown>"))

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                end_time = time.perf_counter()
                duration = end_time - start_time
                self.times.append(duration)
                self._name_to_times.setdefault(label, []).append(duration)

        return wrapper

    # Summary/stat helpers
    def get_total_time(self):
        return sum(self.times)

    def get_max_time(self):
        return max(self.times) if self.times else 0.0

    def get_min_time(self):
        return min(self.times) if self.times else 0.0

    def get_function_stats(self):
        stats = {}
        for name, times in self._name_to_times.items():
            if not times:
                continue
            total = sum(times)
            count = len(times)
            stats[name] = {
                "count": count,
                "total": total,
                "avg": total / count,
                "fps": count / total if total > 0 else 0.0,
                "min": min(times),
                "max": max(times),
            }
        return stats

    # --- Manual code block timing ---
    @contextmanager
    def section(self, name):
        start_time = time.perf_counter()
        try:
            yield
        finally:
            end_time = time.perf_counter()
            duration = end_time - start_time
            self.times.append(duration)
            self._name_to_times.setdefault(name, []).append(duration)

    def start(self, name):
        if not hasattr(self, "_running"):
            self._running = {}
        # Keep latest start; overwrite silently
        self._running[name] = time.perf_counter()

    def stop(self, name):
        if not hasattr(self, "_running"):
            return 0.0
        start_time = self._running.pop(name, None)
        if start_time is None:
            return 0.0
        duration = time.perf_counter() - start_time
        self.times.append(duration)
        self._name_to_times.setdefault(name, []).append(duration)
        return duration

    def _report_summary(self, save_dir=None):
        if not self._name_to_times:
            return
        stats = self.get_function_stats()
        lines = ["[Timer] Summary (auto) – per decorated function:"]
        # Sort by total time desc
        for name, s in sorted(stats.items(), key=lambda kv: kv[1]["total"], reverse=True):
            lines.append(
                f"  - {name}: count={s['count']}, total={s['total']:.6f}s, "
                f"avg={s['avg']:.6f}s, min={s['min']:.6f}s, max={s['max']:.6f}s, fps={s['fps']:.2f}"
            )
        total_count = stats.get("Tracking", {}).get("count", 0)
        total_time = self.get_total_time()
        if total_count > 0 and total_time > 0:
            lines.append(
                f"[Timer] Overall: count={total_count}, total={total_time:.6f}s, "
                f"avg={total_time/total_count:.6f}s, fps={total_count/total_time:.2f}"
            )
        self._report_fn("\n".join(lines))
        if save_dir:
            self._write_csv(save_dir, stats, total_time, total_count)

    def _write_csv(self, save_dir, stats, total_time, total_count):
        try:
            csv_path = f"{save_dir}/timer_summary.csv"
            existing_rows = {}
            # read and keep existing rows if file exists
            if os.path.exists(csv_path):
                try:
                    with open(csv_path, "r") as f:
                        lines = [line.strip() for line in f.readlines()]
                    for line in lines[1:]:  # skip header
                        if not line:
                            continue
                        parts = line.split(",")
                        if len(parts) < 5:
                            continue
                        name = parts[0]
                        existing_rows[name] = line
                except Exception:
                    existing_rows = {}

            # remove overlapping names (will be replaced by current stats)
            for name in list(stats.keys()) + ["Full System"]:
                if name in existing_rows:
                    del existing_rows[name]

            with open(csv_path, "w") as file:
                file.write("Name,Count,Total Time,Average Time,FPS\n")
                # write remaining existing rows first (to keep stable ordering for older entries)
                for _, line in existing_rows.items():
                    file.write(line + "\n")
                # now write current stats
                for name, s in stats.items():
                    file.write(
                        f"{name},{s['count']},{s['total']:.6f},{s['avg']:.6f},{s['fps']:.2f}\n"
                    )
                if total_count > 0 and total_time > 0:
                    file.write(
                        f"Full System,{total_count},{total_time:.6f},{total_time/total_count:.6f},{total_count/total_time:.2f}\n"
                    )
        except Exception:
            pass

# A convenient global instance for simple usage: from sys_timer import timer
timer = Timer(auto_report=False)