"""The `jolto` command: the dispatcher, `python -m jolto` and every subcommand's --help.

No market data needed; --help exits before any input is read.
"""

import subprocess
import sys

from jolto.cli import COMMANDS, main


def _run(*args):
    return subprocess.run([sys.executable, "-m", "jolto", *args], capture_output=True, text=True)


def test_dispatcher():
    assert main([]) == 0 and main(["--help"]) == 0
    assert main(["no-such-command"]) == 2


def test_subcommand_help():
    r = _run("--help")
    assert r.returncode == 0 and all(name in r.stdout for name in COMMANDS), r.stdout
    for name in COMMANDS:
        r = _run(name, "--help")
        assert r.returncode == 0 and f"usage: jolto {name}" in r.stdout, (name, r.stdout, r.stderr)


if __name__ == "__main__":
    test_dispatcher()
    test_subcommand_help()
    print("ALL CHECKS PASSED")
