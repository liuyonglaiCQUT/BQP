"""Export paired ULTRA ranks; audit official metrics before BQP inference."""
import argparse
from collections import Counter, defaultdict, deque
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from easydict import EasyDict
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "repos/ULTRA"), str(ROOT / "adapters")]
from ultra import tasks, util
from ultra.models import Ultra
from bqp_statistics import (paired_statistics, bootstrap_pair_mean,
                            summarize_bootstrap, relation_stratified_permutation)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def relocate(path):
    text = str(path).replace("\\", "/")
    marker = "/model_extensions/"
    if marker not in text:
        raise ValueError("Expected extension-local artifact: " + text)
    result = (ROOT / text.split(marker, 1)[1]).resolve()
    if not result.is_relative_to(ROOT):
        raise ValueError("Path outside extension directory")
    return result


def select_completed_run(directory):
    candidates = []
    for log in directory.rglob("console.log"):
        text = log.read_text(errors="replace")
        if "Evaluate on test" not in text:
            continue
        section = text.rsplit("Evaluate on test", 1)[1]
        metrics = {}
        for metric in ("mrr", "hits@1", "hits@3", "hits@10"):
            found = re.search(r"\b" + re.escape(metric) + r":\s*([0-9.eE+-]+)", section)
            if found:
                metrics[metric] = float(found[1])
        selected = re.findall(r"Load checkpoint from model_epoch_(\d+)\.pth", text)
        if len(metrics) == 4 and selected:
            candidates.append((log.stat().st_mtime, log, int(selected[-1]), metrics))
    if not candidates:
        raise ValueError("No completed finetuning log with selected checkpoint and test metrics.")
    _, log, epoch, metrics = max(candidates, key=lambda item: item[0])
    config = log.parent / "config.yaml"
    cfg = EasyDict(yaml.safe_load(config.read_text()))
    matches = list(relocate(cfg.output_dir).rglob(f"model_epoch_{epoch}.pth"))
    if len(matches) != 1:
        raise ValueError(f"Expected one selected checkpoint, found {matches}")
    return cfg, matches[0], log, metrics


def rank_versions(pred, target, mask):
    if not torch.isfinite(pred).all():
        raise ValueError("Nonfinite model scores")
    truth = pred.gather(1, target[:, None])
    greater = ((pred > truth) & mask).sum(1)
    ties = ((pred == truth) & mask).sum(1)
    official = tasks.compute_ranking(pred, target, mask)
    assert torch.equal(official, greater + ties + 1)
    average = 1 + greater.double() + ties.double() / 2
    return official.cpu().numpy(), average.cpu().numpy()


def metrics(ranks):
    return {"mrr": float(np.mean(1 / ranks)),
            **{f"hits@{k}": float(np.mean(ranks <= k)) for k in (1, 3, 10)}}


def verify_raw_identity(dataset, dataset_name):
    """Reproduce loader mappings and align raw test rows to processed test order."""
    raw = {}
    for path in map(Path, dataset.raw_paths):
        raw[path.name] = path
    split_rows = {
        split: [line.split() for line in raw[split + ".txt"].read_text().splitlines()
                if line.strip()]
        for split in ("train", "valid", "test")
    }
    if dataset_name == "WN18RR":
        # PyG WordNet18RR assigns unseen IDs by split, all heads then all tails,
        # and subsequently sorts all edges by (head, tail). Reproduce both facts.
        entity_map = {}
        for split in ("train", "valid", "test"):
            rows = split_rows[split]
            for label in [row[0] for row in rows] + [row[2] for row in rows]:
                entity_map.setdefault(label, len(entity_map))
        relation_map = dict(dataset.edge2id)
        ordering = "PyG global (head,tail) sort; raw rows realigned by exact h,t,r ID"
    else:
        def read_dictionary(path):
            result = {}
            for line in path.read_text().splitlines():
                if line.strip():
                    index, label = line.split("\t")
                    result[label] = int(index)
            return result
        entity_map = read_dictionary(raw["entities.dict"])
        relation_map = read_dictionary(raw["relations.dict"])
        ordering = "PyG raw split order; audited and realigned by exact h,t,r ID"

    test_rows = None
    hashes = {}
    for i, split in enumerate(("train", "valid", "test")):
        path = raw[split + ".txt"]
        rows = split_rows[split]
        graph = dataset[i]
        ids = torch.cat([graph.target_edge_index.t(),
                         graph.target_edge_type[:, None]], 1).cpu().numpy()
        if len(rows) != len(ids):
            raise ValueError(f"{split}: raw and processed count mismatch")
        encoded = np.asarray([
            [entity_map[row[0]], entity_map[row[2]], relation_map[row[1]]]
            for row in rows
        ], dtype=np.int64)
        if Counter(map(tuple, encoded)) != Counter(map(tuple, ids)):
            raise ValueError(f"{split}: raw/processed triple multiset mismatch")
        hashes[split] = digest(path)
        if split == "test":
            buckets = defaultdict(deque)
            for row, triple in zip(rows, encoded):
                buckets[tuple(triple)].append(row)
            aligned = []
            for triple in ids:
                key = tuple(triple)
                if not buckets[key]:
                    raise ValueError("Processed test triple lacks raw source identity")
                aligned.append(buckets[key].popleft())
            if any(values for values in buckets.values()):
                raise ValueError("Raw test triple was not consumed exactly once")
            test_rows = np.asarray(aligned, dtype=str)
    expected = 3134 if dataset_name == "WN18RR" else 20466
    assert len(test_rows) == expected
    return test_rows, hashes, {
        "method": ordering,
        "entity_labels": len(entity_map),
        "relation_labels": len(relation_map),
        "multiset_and_multiplicity_checked": True,
    }


def analyze(ranks, relations, repeats, seed):
    head, tail = 1 / ranks[:, 0], 1 / ranks[:, 1]
    independent, cluster = bootstrap_pair_mean(tail, head, repeats, np.random.default_rng(seed))
    null, permutation = relation_stratified_permutation(
        tail, head, relations, repeats, np.random.default_rng(seed + 1))
    return {"analytic": paired_statistics(tail, head),
            "bootstrap": summarize_bootstrap(independent, cluster),
            "within_relation_repairing": permutation}, {
                "bootstrap_independent": independent, "bootstrap_cluster": cluster,
                "within_relation_null_correlation": null}


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset", choices=["WN18RR", "FB15k237"])
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--repeats", type=int, default=2000)
    a = p.parse_args()
    if a.repeats < 100:
        raise ValueError("Use at least 100 repeats")
    run = ROOT / "runs/ULTRA" / a.dataset / f"finetune_seed_{a.seed}"
    cfg, checkpoint, log, reference = select_completed_run(run)
    out = ROOT / "results/BQP_ULTRA" / a.dataset / f"seed_{a.seed}" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True)
    print(f"Selected from validation-selection log: {checkpoint}", flush=True)
    cfg.dataset.root = str(ROOT / "data/ULTRA")
    dataset = util.build_dataset(cfg)
    raw_rows, hashes, identity_audit = verify_raw_identity(dataset, a.dataset)
    device = torch.device("cuda:0")
    torch.manual_seed(a.seed)
    graph = dataset[2].to(device)
    known = Data(edge_index=dataset._data.target_edge_index,
                 edge_type=dataset._data.target_edge_type,
                 num_nodes=graph.num_nodes).to(device)
    model = Ultra(rel_model_cfg=cfg.model.relation_model,
                  entity_model_cfg=cfg.model.entity_model).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device)["model"], strict=True)
    model.eval()
    triples = torch.cat([graph.target_edge_index.t(), graph.target_edge_type[:, None]], 1)
    official, average = [], []
    for begin in range(0, len(triples), cfg.train.batch_size):
        batch = triples[begin:begin + cfg.train.batch_size]
        tail_batch, head_batch = tasks.all_negative(graph, batch)
        tail_mask, head_mask = tasks.strict_negative_mask(known, batch)
        tr, ta = rank_versions(model(graph, tail_batch), batch[:, 1], tail_mask)
        hr, ha = rank_versions(model(graph, head_batch), batch[:, 0], head_mask)
        official.append(np.stack([hr, tr], 1))
        average.append(np.stack([ha, ta], 1))
        if begin % (cfg.train.batch_size * 100) == 0:
            print(f"Export {begin}/{len(triples)} triples", flush=True)
    official, average = np.concatenate(official).astype(float), np.concatenate(average)
    actual = metrics(official)
    drift = {key: abs(actual[key] - reference[key]) for key in reference}
    passed = max(drift.values()) <= 2e-6
    ids = triples.cpu().numpy()  # Explicit column order: h,t,r
    np.savez_compressed(out / "test_triple_ranks.npz", ranks_official=official,
                        ranks_average=average, raw_triples_hrt=raw_rows,
                        triples_htr=ids, source_triple_ids=np.arange(len(ids)),
                        directions=np.array(["head", "tail"]))
    report = {"dataset": a.dataset, "seed": a.seed, "checkpoint": str(checkpoint),
              "checkpoint_sha256": digest(checkpoint), "selection_log": str(log),
              "raw_split_sha256": hashes, "filter": "train+valid+test",
              "source_identity_audit": identity_audit,
              "source_triples": len(ids), "official_metrics": actual,
              "logged_metrics": reference, "metric_abs_drift": drift,
              "audit_passed": passed, "average_tie_metrics": metrics(average),
              "scope": "Single fixed checkpoint; source-triple sampling uncertainty conditional on this graph/model. Not training-seed uncertainty; cross-triple dependence not resolved."}
    (out / "audit.json").write_text(json.dumps(report, indent=2))
    if not passed:
        raise RuntimeError(f"Official metric reproduction failed; stop BQP inference. See {out}")
    results = {}
    for name, ranks in (("official_pessimistic", official), ("harmonized_average", average)):
        results[name], samples = analyze(ranks, ids[:, 2], a.repeats, a.seed)
        np.savez_compressed(out / f"{name}_resampling.npz", **samples)
    (out / "bqp_report.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({"output": str(out), "audit_passed": True, "BQP": results}, indent=2), flush=True)


if __name__ == "__main__":
    main()
