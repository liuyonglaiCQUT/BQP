"""Focused offline checks for export ranks, pairing and BQP calculations."""
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if os.name == "nt":
    sys.path.insert(0, str(ROOT / "local_packages"))
import numpy as np
import torch
from validate_ultra_bqp import rank_versions, analyze, verify_raw_identity
from torch_geometric.data import Data
import tempfile

scores = torch.tensor([[3., 3., 5., 1.], [1., 2., 3., 4.]])
targets = torch.tensor([0, 2])
mask = torch.tensor([[False, True, False, True], [True, True, False, True]])
official, average = rank_versions(scores, targets, mask)
np.testing.assert_array_equal(official, [2, 2])
np.testing.assert_array_equal(average, [1.5, 2])
# Every source has two identical outcomes: ignoring pairing reduces SE by sqrt(2).
ranks = np.stack([np.arange(1, 101), np.arange(1, 101)], axis=1).astype(float)
report, _ = analyze(ranks, np.zeros(100), 200, 7)
assert abs(report["analytic"]["se_ratio_cluster_over_independent"] - np.sqrt(2)) < 1e-12
assert report["within_relation_repairing"]["p_value_greater"] < .02
# Verify raw label identity check and detect shuffled/incorrect loader identities.
(ROOT / "tmp").mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as folder:
    class Dataset:
        edge2id = {"r": 0}
        def __getitem__(self, i):
            return self.graphs[i]
    d = Dataset()
    d.raw_paths, d.graphs = [], []
    for split, n in (("train", 2), ("valid", 2), ("test", 3134)):
        path = Path(folder) / (split + ".txt")
        path.write_text("a r b\n" * n)
        d.raw_paths.append(str(path))
        d.graphs.append(Data(target_edge_index=torch.tensor([[0] * n, [1] * n]),
                             target_edge_type=torch.zeros(n, dtype=torch.long)))
    rows, _, audit = verify_raw_identity(d, "WN18RR")
    assert rows.shape == (3134, 3)
    assert audit["multiset_and_multiplicity_checked"]
    d.graphs[2].target_edge_index[0, 0] = 2
    try:
        verify_raw_identity(d, "WN18RR")
    except ValueError:
        pass
    else:
        raise AssertionError("Failed to catch mismatched source identity")
print("PASS: filtered ties, covariance identity, permutation signal, raw source identity guard")
