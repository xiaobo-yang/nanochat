#!/usr/bin/env python3
"""
Create a timestamped experiment directory under the allowed nanochat tmp_exp root.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


ALLOWED_ROOT = Path("/mnt/stepeval/yangxiaobo/cache/nanochat/tmp_exp").resolve()


def sanitize_tag(tag: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in tag.strip())
    cleaned = cleaned.strip("-_")
    if not cleaned:
        raise ValueError("experiment name is empty after sanitization")
    return cleaned


def ensure_allowed(root: Path) -> Path:
    resolved = root.resolve()
    if resolved != ALLOWED_ROOT and ALLOWED_ROOT not in resolved.parents:
        raise ValueError(f"root must be inside {ALLOWED_ROOT}, got {resolved}")
    return resolved


def write_if_missing(path: Path, content: str) -> None:
    if not path.exists():
        path.write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Create experiment directory for nanochat runs")
    parser.add_argument("--name", required=True, help="experiment tag (e.g., moe-topk2-fastpath)")
    parser.add_argument("--root", default=str(ALLOWED_ROOT), help="experiment root path")
    parser.add_argument("--dry-run", action="store_true", help="print target path without writing files")
    args = parser.parse_args()

    root = ensure_allowed(Path(args.root))
    tag = sanitize_tag(args.name)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = root / f"{timestamp}_{tag}"

    if args.dry_run:
        print(run_dir)
        return 0

    run_dir.mkdir(parents=True, exist_ok=False)

    metadata = {
        "name": tag,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    write_if_missing(run_dir / "command.txt", "# Paste the exact training/eval command here.\n")
    write_if_missing(
        run_dir / "notes.md",
        "# Hypothesis\n\n# Change\n\n# Validation Commands\n\n# Results\n\n# Decision\n",
    )
    write_if_missing(run_dir / "metrics.md", "# Metrics\n\n- pending\n")
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
