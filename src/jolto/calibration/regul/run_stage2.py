"""Run Stage 2 with config overrides into results/regul/<tag>/, never touching
production results.

    python -m jolto.calibration.regul.run_stage2 <tag> KEY=VALUE [KEY=VALUE ...]

Values are parsed as Python literals (strings need quotes). Stage-1 parameters are copied from
results/stage1_parameters.csv unless STAGE1=<path> is given.
"""

import ast
import os
import shutil
import sys

from jolto import config
from jolto.calibration.pipeline import run_stage2


def main(argv):
    tag, overrides = argv[0], argv[1:]
    stage1 = os.path.join(config.RESULTS_DIR, config.STAGE1_PARAMS_CSV)
    out = os.path.join(config.RESULTS_DIR, "regul", tag)
    os.makedirs(out, exist_ok=True)
    for kv in overrides:
        key, val = kv.split("=", 1)
        if key == "STAGE1":
            stage1 = val
        else:
            if not hasattr(config, key):
                sys.exit(f"unknown config key {key}")
            setattr(config, key, ast.literal_eval(val))
    shutil.copy(stage1, os.path.join(out, config.STAGE1_PARAMS_CSV))
    config.RESULTS_DIR = out
    run_stage2()


if __name__ == "__main__":
    main(sys.argv[1:])
