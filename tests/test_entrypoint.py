"""Tests for the unique command-line entrypoint."""

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_entrypoint_parses_and_prints_json():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "resihp.train",
            "--config",
            "configs/train.json",
            "--failures",
            "configs/failures.json",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    parsed = json.loads(result.stdout)
    assert parsed["config"]["tp"] == 2
    assert parsed["failures"]["events"][0]["failed_rank"] == 1


def test_old_entrypoint_is_absent():
    assert not (ROOT / "hello_dist.py").exists()
