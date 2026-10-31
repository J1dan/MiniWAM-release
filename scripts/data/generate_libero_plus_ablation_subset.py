#!/usr/bin/env python3
"""Generate or verify the pinned LIBERO-Plus 2K ablation task manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval.libero_plus_protocol import (  # noqa: E402
    load_registry,
    render_manifest,
    select_stratified_tasks,
)


DEFAULT_OUTPUT = PROJECT_ROOT / "scripts" / "eval" / "manifests" / "libero_plus_ablation_2k_seed42.txt"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--libero_plus_root", type=Path, default=PROJECT_ROOT / "LIBERO-plus")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--check", action="store_true",
        help="Verify OUTPUT is byte-identical instead of writing it.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    registry = load_registry(args.libero_plus_root)
    selected, quotas = select_stratified_tasks(registry, count=args.count, seed=args.seed)
    content = render_manifest(selected, quotas, seed=args.seed)
    if args.check:
        if not args.output.is_file() or args.output.read_text(encoding="utf-8") != content:
            raise SystemExit(f"Manifest is missing or stale: {args.output}")
        print(f"Verified {args.output} ({len(selected)} tasks).")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(args.output)
    print(f"Wrote {args.output} ({len(selected)} tasks).")


if __name__ == "__main__":
    main()
