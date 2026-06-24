#!/usr/bin/env python3
"""Build the public `data/` tree from extracted run summaries.

The raw PauliSum snapshots are serialized Julia `.jls` artifacts and are too
large for GitHub.  This script copies the lightweight extracted JSON files from
`runs/` into `data/<system>/...`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
from pathlib import Path
from typing import Any


PUBLIC_FILES = ("config.json", "summary.json", "cdf_layers.json", "yao_overlap.json")
SKIP_TOP_LEVEL = {"_logs", "_monitor"}


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def copy_if_present(src_dir: Path, dst_dir: Path, names: tuple[str, ...]) -> list[str]:
    copied: list[str] = []
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        src = src_dir / name
        if src.is_file():
            shutil.copy2(src, dst_dir / name)
            copied.append(name)
    return copied


def clean_data_dir(data_root: Path) -> None:
    data_root.mkdir(parents=True, exist_ok=True)
    for child in data_root.iterdir():
        if child.name == "README.md":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def build_public_data(
    runs_root: Path,
    data_root: Path,
    *,
    clean: bool,
    source_label: str | None,
) -> dict[str, Any]:
    if clean:
        clean_data_dir(data_root)
    else:
        data_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "generated_at": dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "source": source_label or str(runs_root),
        "layout": "data/<system>/<rule_hash>/{config.json,summary.json,cdf_layers.json}",
        "raw_snapshots": "not included; regenerate into runs/ with scripts/run_one.jl or the original cluster pipeline",
        "systems": {},
    }

    for experiment_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        if experiment_dir.name in SKIP_TOP_LEVEL:
            continue

        # Expected layout: runs/<experiment>/<system>/<rule_hash>/...
        system_dirs = [p for p in experiment_dir.iterdir() if p.is_dir()]
        if not system_dirs:
            continue

        for system_dir in sorted(system_dirs):
            system_name = experiment_dir.name
            public_system_dir = data_root / system_name
            public_system_dir.mkdir(parents=True, exist_ok=True)

            system_index: dict[str, Any] = {
                "system": system_name,
                "source_experiment": experiment_dir.name,
                "source_system_dir": system_dir.name,
                "reference": None,
                "rules": [],
            }

            for run_dir in sorted(p for p in system_dir.iterdir() if p.is_dir()):
                if run_dir.name == "reference":
                    copied = copy_if_present(run_dir, public_system_dir / "reference", PUBLIC_FILES)
                    if copied:
                        system_index["reference"] = {"path": "reference", "files": copied}
                    continue

                copied = copy_if_present(run_dir, public_system_dir / run_dir.name, PUBLIC_FILES)
                if not copied:
                    continue

                config = read_json(run_dir / "config.json")
                summary = read_json(run_dir / "summary.json")
                rule = config.get("rule", {}) if isinstance(config.get("rule", {}), dict) else {}
                system = config.get("system", {}) if isinstance(config.get("system", {}), dict) else {}

                system_index["rules"].append(
                    {
                        "rule_hash": run_dir.name,
                        "label": rule.get("label", run_dir.name),
                        "family": rule.get("family"),
                        "coeff_min": rule.get("coeff_min"),
                        "tau_xy": rule.get("tau_xy"),
                        "tau_xyz": rule.get("tau_xyz"),
                        "n_layers": system.get("n_layers"),
                        "n_complete": summary.get("n_complete"),
                        "complete": summary.get("complete", None),
                        "files": copied,
                    }
                )

            system_index["rules"].sort(key=lambda x: str(x.get("label", "")))
            (public_system_dir / "index.json").write_text(
                json.dumps(system_index, indent=2, sort_keys=True) + "\n"
            )

            total_files = sum(1 for path in public_system_dir.rglob("*") if path.is_file())
            manifest["systems"][system_name] = {
                "path": system_name,
                "n_rules": len(system_index["rules"]),
                "has_reference": system_index["reference"] is not None,
                "n_files": total_files,
            }

    (data_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--source-label", default=None)
    parser.add_argument("--clean", action="store_true", help="remove generated data before rebuilding")
    args = parser.parse_args()

    manifest = build_public_data(
        args.runs_root,
        args.data_root,
        clean=args.clean,
        source_label=args.source_label,
    )
    n_systems = len(manifest["systems"])
    n_rules = sum(system["n_rules"] for system in manifest["systems"].values())
    print(f"wrote {args.data_root} with {n_systems} systems and {n_rules} rule entries")


if __name__ == "__main__":
    main()
