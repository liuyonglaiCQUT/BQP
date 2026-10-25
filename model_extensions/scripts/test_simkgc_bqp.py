"""Focused offline checks for the SimKGC BQP export audit."""
import importlib.util
import tempfile
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).with_name("validate_simkgc_bqp.py")
spec = importlib.util.spec_from_file_location("validate_simkgc_bqp", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

raw = [{"head": "h", "relation": "r", "tail": "t"}]
forward = [{"head": "h", "relation": "r", "tail": "t", "rank": 2}]
backward = [{"head": "t", "relation": "inverse r", "tail": "h", "rank": 3}]
module.validate_pairing(raw, forward, backward)
computed = module.metrics_from_ranks(np.asarray([2, 3]))
assert abs(computed["mrr"] - ((1 / 2 + 1 / 3) / 2)) < 1e-12

metric_file = Path(tempfile.gettempdir()) / "simkgc_bqp_metrics_test.txt"
metric_file.write_text(
    'forward metrics: {"mrr": 0.7223}\n'
    'backward metrics: {"mrr": 0.6274}\n'
    'average metrics: {"mrr": 0.6748}\n',
    encoding="utf-8",
)
logged = module.parse_logged_metrics(metric_file)
assert logged["average"]["mrr"] == 0.6748
assert round((logged["forward"]["mrr"] + logged["backward"]["mrr"]) / 2, 4) == 0.6748
print("SimKGC BQP audit tests passed")
