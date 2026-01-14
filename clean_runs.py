#!/usr/bin/env python

import json
from pathlib import Path
import shutil


def main():
    runs_dir = Path("./runs").resolve()
    for run_dir in runs_dir.iterdir():
        with open(run_dir.joinpath("run_config.json"), "r") as f:
            run_config = json.load(f)
        if run_config["type"] != "test":
            continue
        line_count = 0
        with open(run_dir.joinpath("run_results.csv"), 'r') as f:
            for line_count, _ in enumerate(f, start=1):
                pass  # Just counting here
        if line_count >= 10:
            continue
        shutil.rmtree(run_dir)



if __name__ == "__main__":
    main()