"""The `jolto` command: `jolto <subcommand> [options]`, installed by `pip install -e .`.

Each subcommand reads `data/` and writes `results/` relative to the working directory, with the
settings of jolto/config.py and jolto/val_config.py; `jolto <subcommand> --help` lists its options.
"""

import importlib
import sys

COMMANDS = {
    "calibrate-sabr": ("calibrate_sabr", "Stage 1: fit the rough SABR marginals"),
    "calibrate-correlation": ("calibrate_correlation", "Stage 2: fit the correlation matrix"),
    "simulate-curves": ("simulate_curves", "simulate the forward and discount curves"),
    "price": ("price", "price a product and write its AAD Greeks"),
}


def _usage():
    lines = ["usage: jolto <subcommand> [options]", "", "subcommands:"]
    lines += [f"  {name:<23} {text}" for name, (_, text) in COMMANDS.items()]
    lines += ["", "Run `jolto <subcommand> --help` for its options."]
    return "\n".join(lines)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(_usage())
        return 0
    if argv[0] not in COMMANDS:
        print(f"jolto: unknown subcommand {argv[0]!r}\n\n{_usage()}", file=sys.stderr)
        return 2
    # imported on demand so `jolto --help` stays fast and each subcommand loads only what it needs
    module = importlib.import_module(f"jolto.cli.{COMMANDS[argv[0]][0]}")
    module.main(argv[1:])
    return 0
