#!/usr/bin/env python3
"""Audit downloaded SimKGC and ULTRA BQP result bundles.

The audit is intentionally independent of the training repositories.  It checks
that JSON files parse, NPZ files decompress, expected arrays have the right
shape, and the result metadata identifies the intended dataset/run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

EXPECTED = {
    ("BQP_SimKGC", "WN18RR"): {
        "source_triples": 3134,
        "successful_files": {
            "audit.json",
            "bqp_report.json",
            "resampling.npz",
            "test_triple_ranks.npz",
        },
    },
    ("BQP_ULTRA", "WN18RR"): {
        "source_triples": 3134,
        "successful_files": {
            "audit.json",
            "bqp_report.json",
            "harmonized_average_resampling.npz",
            "official_pessimistic_resampling.npz",
            "test_triple_ranks.npz",
        },
    },
    ("BQP_ULTRA", "FB15k237"): {
        "source_triples": 20466,
        "successful_files": {
            "audit.json",
            "bqp_report.json",
            "harmonized_average_resampling.npz",
            "official_pessimistic_resampling.npz",
            "test_triple_ranks.npz",
        },
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} must contain a JSON object")
    return value


def inspect_npz(path: Path) -> dict[str, Any]:
    arrays: dict[str, Any] = {}
    with np.load(path, allow_pickle=False) as archive:
        for name in archive.files:
            value = archive[name]
            arrays[name] = {"shape": list(value.shape), "dtype": str(value.dtype)}
    return arrays


def validate_rank_archive(
    family: str, arrays: dict[str, Any], source_triples: int
) -> list[str]:
    errors: list[str] = []
    expected_common = {
        "ranks_official": [source_triples, 2],
        "raw_triples_hrt": [source_triples, 3],
        "source_triple_ids": [source_triples],
        "directions": [2],
    }
    for name, shape in expected_common.items():
        actual = arrays.get(name, {}).get("shape")
        if actual != shape:
            errors.append(f"test_triple_ranks.npz:{name} shape {actual}, expected {shape}")
    if family == "BQP_ULTRA":
        actual = arrays.get("ranks_average", {}).get("shape")
        if actual != [source_triples, 2]:
            errors.append(
                "test_triple_ranks.npz:ranks_average shape "
                f"{actual}, expected {[source_triples, 2]}"
            )
    return errors


def validate_resampling_archive(path: Path, arrays: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not arrays:
        return [f"{path.name} contains no arrays"]
    lengths = {meta["shape"][0] for meta in arrays.values() if meta["shape"]}
    if lengths != {2000}:
        errors.append(f"{path.name} resample lengths are {sorted(lengths)}, expected [2000]")
    for name, meta in arrays.items():
        if meta["shape"] != [2000]:
            errors.append(f"{path.name}:{name} shape {meta['shape']}, expected [2000]")
    return errors


def inspect_run(run_dir: Path, family: str, dataset: str) -> dict[str, Any]:
    spec = EXPECTED[(family, dataset)]
    expected_n = int(spec["source_triples"])
    errors: list[str] = []
    files: dict[str, Any] = {}
    parsed_json: dict[str, dict[str, Any]] = {}
    parsed_npz: dict[str, dict[str, Any]] = {}

    for path in sorted(p for p in run_dir.iterdir() if p.is_file()):
        item: dict[str, Any] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        try:
            if path.suffix == ".json":
                parsed_json[path.name] = read_json(path)
                item["parse"] = "ok"
            elif path.suffix == ".npz":
                parsed_npz[path.name] = inspect_npz(path)
                item["parse"] = "ok"
                item["arrays"] = parsed_npz[path.name]
        except Exception as exc:  # report every damaged artifact in one pass
            item["parse"] = "failed"
            item["error"] = f"{type(exc).__name__}: {exc}"
            errors.append(f"{path.name} cannot be read: {exc}")
        files[path.name] = item

    audit = parsed_json.get("audit.json", {})
    audit_passed = audit.get("audit_passed") is True
    if not audit:
        errors.append("missing or unreadable audit.json")
    if audit and audit.get("dataset") != dataset:
        errors.append(f"audit dataset is {audit.get('dataset')!r}, expected {dataset!r}")
    if audit and int(audit.get("source_triples", -1)) != expected_n:
        errors.append(
            f"audit source_triples is {audit.get('source_triples')}, expected {expected_n}"
        )

    present = set(files)
    missing = sorted(set(spec["successful_files"]) - present)
    if audit_passed and missing:
        errors.append(f"successful run is missing files: {', '.join(missing)}")

    ranks = parsed_npz.get("test_triple_ranks.npz")
    if ranks is not None:
        errors.extend(validate_rank_archive(family, ranks, expected_n))

    for name, arrays in parsed_npz.items():
        if "resampling" in name:
            errors.extend(validate_resampling_archive(Path(name), arrays))

    if audit_passed:
        status = "complete_success" if not errors else "invalid_success"
    else:
        # A stopped run may legitimately have only audit + ranks.  It is retained
        # as provenance but must never be selected for paper analysis.
        status = "failed_audit_artifact" if not errors else "invalid_failed_artifact"

    summary: dict[str, Any] = {
        "path": str(run_dir.relative_to(ROOT)),
        "family": family,
        "dataset": dataset,
        "seed": run_dir.parent.name.removeprefix("seed_"),
        "timestamp": run_dir.name,
        "audit_passed": audit_passed,
        "status": status,
        "missing_success_files": missing,
        "errors": errors,
        "files": files,
    }
    if audit:
        summary["checkpoint"] = audit.get("checkpoint")
        summary["checkpoint_sha256"] = audit.get("checkpoint_sha256")
        summary["source_triples"] = audit.get("source_triples")
        summary["official_metrics"] = audit.get(
            "official_metrics",
            audit.get("logged_metrics", {}).get("average"),
        )
    report = parsed_json.get("bqp_report.json", {})
    if family == "BQP_SimKGC" and report:
        summary["bqp_key_metrics"] = report.get("analytic")
    elif family == "BQP_ULTRA" and report:
        summary["bqp_key_metrics"] = report.get("official_pessimistic", {}).get(
            "analytic"
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "bqp_download_integrity.json",
    )
    args = parser.parse_args()

    runs: list[dict[str, Any]] = []
    missing_expected_roots: list[str] = []
    for (family, dataset), _spec in EXPECTED.items():
        dataset_root = ROOT / "results" / family / dataset
        if not dataset_root.is_dir():
            missing_expected_roots.append(str(dataset_root.relative_to(ROOT)))
            continue
        timestamp_dirs = sorted(dataset_root.glob("seed_*/*"))
        if not timestamp_dirs:
            missing_expected_roots.append(str(dataset_root.relative_to(ROOT)))
        for run_dir in timestamp_dirs:
            if run_dir.is_dir():
                runs.append(inspect_run(run_dir, family, dataset))

    successful = [run for run in runs if run["status"] == "complete_success"]
    successful_pairs = {(run["family"], run["dataset"]) for run in successful}
    expected_pairs = set(EXPECTED)
    overall_passed = (
        not missing_expected_roots
        and expected_pairs.issubset(successful_pairs)
        and not any(run["status"] == "invalid_success" for run in runs)
    )
    report = {
        "root": str(ROOT),
        "overall_passed": overall_passed,
        "expected_successful_result_sets": len(EXPECTED),
        "complete_successful_result_sets": len(successful),
        "missing_expected_roots": missing_expected_roots,
        "note": (
            "failed_audit_artifact is preserved provenance and is excluded from "
            "paper analysis; it does not make a later successful result incomplete."
        ),
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    print(json.dumps({
        "overall_passed": overall_passed,
        "report": str(args.output),
        "runs": [
            {
                "family": run["family"],
                "dataset": run["dataset"],
                "timestamp": run["timestamp"],
                "status": run["status"],
                "errors": run["errors"],
            }
            for run in runs
        ],
    }, ensure_ascii=False, indent=2))
    if not overall_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
