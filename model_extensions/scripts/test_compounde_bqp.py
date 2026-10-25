"""Focused offline checks for CompoundE checkpoint selection and BQP analysis."""
import importlib.util
import json
from pathlib import Path
import tempfile

import numpy as np
import sys

SCRIPT = Path(__file__).with_name("validate_compounde_bqp.py")
spec = importlib.util.spec_from_file_location("validate_compounde_bqp", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_compounde

trajectory = Path(tempfile.gettempdir()) / "compounde_validation_trajectory.jsonl"
trajectory.write_text("\n".join([
    json.dumps({"step": 5000, "mrr": 0.2, "selected": True}),
    json.dumps({"step": 10000, "mrr": 0.3, "selected": True}),
    json.dumps({"step": 15000, "mrr": 0.3, "selected": False}),
]) + "\n", encoding="utf-8")
step, score, rows = module.trajectory_selection(trajectory)
assert (step, score, len(rows)) == (10000, 0.3, 3)

tail = np.asarray([1.0, 0.5, 0.25, 0.125] * 50)
head = tail.copy()
ranks = np.stack([1.0 / head, 1.0 / tail], axis=1)
relations = np.asarray([0, 0, 1, 1] * 50)
report, samples = module.analyze(ranks, relations, repeats=200, seed=7)
assert report["analytic"]["head_tail_correlation"] > 0.999
assert report["analytic"]["se_ratio_cluster_over_independent"] > 1.0
assert len(samples["bootstrap_cluster"]) == 200

computed = module.metrics_from_ranks(np.asarray([[1.0, 2.0], [4.0, 8.0]]))
assert abs(computed["mrr"] - np.mean([1.0, 0.5, 0.25, 0.125])) < 1e-12

train = np.asarray([[0, 0, 1], [0, 0, 2], [3, 0, 1]], dtype=np.int64)
training_filters = run_compounde.filters(train)
sampled = run_compounde.sample_negatives(
    train, "tail", 5, 100, training_filters, np.random.default_rng(7), "filtered")
assert not np.isin(sampled[0], [1, 2]).any()
assert not np.isin(sampled[2], [1]).any()
print("CompoundE BQP audit tests passed")
