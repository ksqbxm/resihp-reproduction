"""Training entrypoint placeholder for the ResiHP implementation."""

import argparse
import json
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="ResiHP training entrypoint")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--failures", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    failures = json.loads(args.failures.read_text(encoding="utf-8"))
    print(json.dumps({"config": config, "failures": failures}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
