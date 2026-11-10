import os
import sys


DATA_HINT = (
    "Market data is not distributed with the repository; data/README.md lists the expected "
    "files and their schema."
)


def check_config(module, names, script):
    """Entry-point guard: exit 1 listing the names `module` lacks (the script's CONFIG CONTRACT)."""
    missing = [name for name in names if not hasattr(module, name)]
    if missing:
        print(f"ERROR: {module.__name__} is missing entries that {script} needs:")
        for name in missing:
            print(f"   • {name}")
        print(f"\nAdd them and re-run. The CONFIG CONTRACT block of {script} gives their types.")
        sys.exit(1)


def require_files(paths, hint=DATA_HINT):
    """Entry-point guard: exit 1 with `hint` when an input file or directory is missing."""
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        print("ERROR: input not found:")
        for p in missing:
            print(f"   • {p}")
        print(f"\n{hint}")
        sys.exit(1)


def print_header(title):
    print("\n" + "=" * 60)
    print(f"{title:^60}")
    print("=" * 60)


def log_progress(category, message, level=0):
    indent = "   " * level
    if level == 2:
        print(f"{indent}> {message}")
    else:
        print(f"{indent}[{category}] {message}")


def print_greek_ladder(deltas, vegas, tenors):
    print("\n" + "=" * 50)
    print(f"{'GREEK LADDERS (PV01/Vega in bps)':^50}")
    print("-" * 50)
    print(f"{'Tenor':<10} | {'Delta (bps)':>15} | {'Vega (bps)':>15}")
    print("-" * 50)
    for i, t in enumerate(tenors):
        delta, vega = deltas[i], vegas[i]
        # skip negligible risk
        if abs(delta) > 1e-7 or abs(vega) > 1e-7:
            print(f"{t:4.1f}Y      | {delta:15.4f} | {vega:15.4f}")
    print("-" * 50)


class DualLogger:
    """Tees stdout (+ stderr, in context-manager mode) to a utf-8 log file.

    Serves two call styles against the single ``__init__`` tee (``self.terminal`` / ``self.log``):

    * File-like (stage2.py): ``sys.stdout = DualLogger(path)`` then ``sys.stdout.close()``,
      driving the tee through ``write``/``flush``/``close``.
    * Context manager (jolto calibrate-sabr): ``with DualLogger(path):``. ``__enter__`` redirects
      stdout/stderr through ``self``, ``__exit__`` restores them and closes the log.
    """

    def __init__(self, filepath):
        dir_name = os.path.dirname(filepath)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        self.terminal = sys.stdout
        self.log = open(filepath, "a", encoding="utf-8")

    def __enter__(self):
        self._old_out, self._old_err = sys.stdout, sys.stderr
        # re-capture the stdout live at entry (it may differ from the one seen at __init__);
        # write()/flush() fan out to self.terminal and self.log
        self.terminal = self._old_out
        sys.stdout = self
        sys.stderr = self
        return self

    def __exit__(self, *_):
        sys.stdout = self._old_out
        sys.stderr = self._old_err
        self.flush()
        self.close()
        return False

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()  # flush immediately in case of a crash

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def safe_cpu_compile(func):
    import torch
    import warnings
    from jolto import config

    if not config.USE_TORCH_COMPILE:
        return func

    try:
        import torch._dynamo

        torch._dynamo.config.suppress_errors = True
        return torch.compile(func, dynamic=True)
    except Exception as e:
        warnings.warn(f"PyTorch compiler bypassed: {e}")
        return func
