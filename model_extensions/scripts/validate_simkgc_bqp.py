"""Audit official SimKGC rank exports and run paired BQP inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "adapters"))
from bqp_statistics import (bootstrap_pair_mean, paired_statistics,
                            relation_stratified_permutation,
                            summarize_bootstrap)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def verify_nell995_protocol(raw: list[dict], raw_path: Path) -> dict:
    data_dir = raw_path.parent
    manifest_path = data_dir / "adapter_manifest.json"
    test_txt = data_dir / "test.txt"
    manifest = load_json(manifest_path)
    if manifest.get("format") != "simkgc-nell995-bqp-adapter-v1":
        raise ValueError("Unexpected NELL995 adapter manifest format")
    output_hashes = manifest.get("output_files_sha256", {})
    for path in (raw_path, test_txt):
        actual = digest(path)
        expected = output_hashes.get(path.name)
        if actual != expected:
            raise ValueError("NELL995 adapter hash mismatch for {}: {} != {}".format(
                path, actual, expected))
    text_triples = []
    for line_number, line in enumerate(test_txt.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split("\t")
        if len(fields) != 3:
            raise ValueError("Malformed NELL995 test triple at row {}".format(line_number))
        text_triples.append(tuple(fields))
    json_triples = [(row["head_id"], row["relation"], row["tail_id"]) for row in raw]
    if text_triples != json_triples:
        raise ValueError("NELL995 SimKGC JSON does not preserve audited test row identity/order")
    if len(json_triples) != 3992 or manifest.get("source_split_sizes", {}).get("test") != 3992:
        raise ValueError("Unexpected NELL995 test size")
    return {
        "adapter_manifest_sha256": digest(manifest_path),
        "source_derived_split_seed": manifest.get("source_derived_split_seed"),
        "source_split_sizes": manifest.get("source_split_sizes"),
        "source_output_files_sha256": manifest.get("source_output_files_sha256"),
        "simkgc_test_txt_sha256": digest(test_txt),
        "json_test_identity_and_order_match_txt": True,
    }


def metrics_from_ranks(ranks: np.ndarray) -> dict[str, float]:
    ranks = np.asarray(ranks, dtype=np.float64)
    return {
        "mean_rank": float(ranks.mean()),
        "mrr": float(np.mean(1.0 / ranks)),
        "hit@1": float(np.mean(ranks <= 1)),
        "hit@3": float(np.mean(ranks <= 3)),
        "hit@10": float(np.mean(ranks <= 10)),
    }


def parse_logged_metrics(path: Path) -> dict[str, dict[str, float]]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        for name in ("forward", "backward", "average"):
            prefix = name + " metrics: "
            if line.startswith(prefix):
                result[name] = {key: float(value) for key, value in
                                json.loads(line[len(prefix):]).items()}
    if set(result) != {"forward", "backward", "average"}:
        raise ValueError("Missing directional metrics in {}: {}".format(
            path, sorted(result)))
    return result


def validate_pairing(raw: list[dict], forward: list[dict], backward: list[dict]) -> None:
    if not (len(raw) == len(forward) == len(backward)):
        raise ValueError("Raw/forward/backward counts differ: {}/{}/{}".format(
            len(raw), len(forward), len(backward)))
    for index, (source, tail_query, head_query) in enumerate(zip(raw, forward, backward)):
        expected_tail = (source["head"], source["relation"], source["tail"])
        actual_tail = (tail_query["head"], tail_query["relation"], tail_query["tail"])
        expected_head = (source["tail"], "inverse " + source["relation"], source["head"])
        actual_head = (head_query["head"], head_query["relation"], head_query["tail"])
        if actual_tail != expected_tail:
            raise ValueError("Forward/source mismatch at row {}".format(index))
        if actual_head != expected_head:
            raise ValueError("Backward/source mismatch at row {}".format(index))
        for direction, item in (("forward", tail_query), ("backward", head_query)):
            rank = item.get("rank")
            if not isinstance(rank, int) or rank < 1:
                raise ValueError("Missing/invalid rank at {} row {}: {}".format(
                    direction, index, rank))


def analyze(head_ranks: np.ndarray, tail_ranks: np.ndarray,
            relations: np.ndarray, repeats: int, seed: int):
    head = 1.0 / head_ranks
    tail = 1.0 / tail_ranks
    independent, cluster = bootstrap_pair_mean(
        tail, head, repeats, np.random.default_rng(seed))
    null, permutation = relation_stratified_permutation(
        tail, head, relations, repeats, np.random.default_rng(seed + 1))
    report = {
        "analytic": paired_statistics(tail, head),
        "bootstrap": summarize_bootstrap(independent, cluster),
        "within_relation_repairing": permutation,
    }
    samples = {
        "bootstrap_independent": independent,
        "bootstrap_cluster": cluster,
        "within_relation_null_correlation": null,
    }
    return report, samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["WN18RR", "FB15k237", "NELL995"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--repeats", type=int, default=2000)
    args = parser.parse_args()
    if args.repeats < 100:
        raise ValueError("Use at least 100 repeats")

    run = ROOT / "runs" / "SimKGC" / args.dataset / "seed_{}".format(args.seed)
    checkpoint = run / "model_last.mdl"
    raw_path = ROOT / "repos" / "SimKGC" / "data" / args.dataset / "test.txt.json"
    forward_path = run / "eval_test.txt.json_forward_model_last.mdl.json"
    backward_path = run / "eval_test.txt.json_backward_model_last.mdl.json"
    metrics_path = run / "metrics_test.txt.json_model_last.mdl.json"
    for path in (checkpoint, raw_path, forward_path, backward_path, metrics_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    raw = load_json(raw_path)
    dataset_protocol = (verify_nell995_protocol(raw, raw_path)
                        if args.dataset == "NELL995" else None)
    forward = load_json(forward_path)
    backward = load_json(backward_path)
    validate_pairing(raw, forward, backward)
    head_ranks = np.asarray([item["rank"] for item in backward], dtype=np.int64)
    tail_ranks = np.asarray([item["rank"] for item in forward], dtype=np.int64)
    actual = {
        "forward": metrics_from_ranks(tail_ranks),
        "backward": metrics_from_ranks(head_ranks),
    }
    rounded = {
        direction: {key: round(value, 4) for key, value in values.items()}
        for direction, values in actual.items()
    }
    # SimKGC rounds each direction to four decimals before averaging them.
    rounded["average"] = {
        key: round((rounded["forward"][key] + rounded["backward"][key]) / 2.0, 4)
        for key in rounded["forward"]
    }
    actual["pooled_unrounded"] = metrics_from_ranks(
        np.concatenate([head_ranks, tail_ranks]))
    logged = parse_logged_metrics(metrics_path)
    drift = {
        direction: {key: abs(rounded[direction][key] - logged[direction][key])
                    for key in logged[direction]}
        for direction in ("forward", "backward", "average")
    }
    passed = all(value <= 1e-12 for values in drift.values() for value in values.values())

    output = (ROOT / "results" / "BQP_SimKGC" / args.dataset /
              "seed_{}".format(args.seed) /
              datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(parents=True)
    ids = np.asarray([[row["head_id"], row["relation"], row["tail_id"]]
                      for row in raw], dtype=str)
    ranks = np.stack([head_ranks, tail_ranks], axis=1)
    np.savez_compressed(
        output / "test_triple_ranks.npz",
        ranks_official=ranks,
        raw_triples_hrt=ids,
        source_triple_ids=np.arange(len(raw)),
        directions=np.asarray(["head", "tail"]),
    )
    audit = {
        "model": "SimKGC",
        "dataset": args.dataset,
        "seed": args.seed,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest(checkpoint),
        "raw_test_sha256": digest(raw_path),
        "dataset_protocol": dataset_protocol,
        "forward_export_sha256": digest(forward_path),
        "backward_export_sha256": digest(backward_path),
        "source_triples": len(raw),
        "directional_queries": 2 * len(raw),
        "pairing_audit": "exact ordered name match; backward=(tail, inverse relation, head)",
        "filter": "official SimKGC train+valid+test filter",
        "ranking": "official descending torch.sort position after graph reranking",
        "recomputed_metrics": actual,
        "recomputed_official_rounding": rounded,
        "logged_metrics": logged,
        "metric_abs_drift_after_official_rounding": drift,
        "audit_passed": passed,
        "scope": ("Single fixed checkpoint; source-triple sampling uncertainty conditional "
                  "on this graph/model. Not training-seed uncertainty; cross-triple "
                  "dependence is not resolved."),
    }
    (output / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    if not passed:
        raise RuntimeError("Official metric audit failed; stop BQP inference: {}".format(output))

    relations = np.asarray([row["relation"] for row in raw])
    report, samples = analyze(head_ranks, tail_ranks, relations, args.repeats, args.seed)
    np.savez_compressed(output / "resampling.npz", **samples)
    (output / "bqp_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "audit_passed": True,
                      "official_metrics": rounded["average"],
                      "pooled_unrounded_metrics": actual["pooled_unrounded"],
                      "BQP": report}, indent=2))


if __name__ == "__main__":
    main()
