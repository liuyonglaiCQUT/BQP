"""Resume single-GPU ULTRA at a saved epoch boundary, retaining validation selection."""
import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import json
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("dataset", choices=["WN18RR", "FB15k237"])
p.add_argument("--seed", type=int, default=7)
a = p.parse_args()
original = ROOT / "runs" / "ULTRA" / a.dataset / f"finetune_seed_{a.seed}"
cfg = yaml.safe_load((original / "config.yaml").read_text())
history = {}
for path in sorted(original.rglob("model_epoch_*.pth"), key=lambda x: x.stat().st_mtime):
    epoch = int(path.stem.rsplit("_", 1)[1])
    try:
        state = torch.load(path, map_location="cpu")
        if not {"model", "optimizer"} <= state.keys():
            raise ValueError("Missing model or optimizer")
        history[epoch] = str(path.resolve())
    except Exception as exc:
        print(f"Skipping unreadable checkpoint {path}: {exc}", flush=True)
if not history:
    raise SystemExit("No readable epoch checkpoint; cannot resume.")
epoch = max(history)
if epoch > cfg["train"]["num_epoch"]:
    raise SystemExit("Checkpoint epoch exceeds configured training budget.")
out = original / "resumptions" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
out.mkdir(parents=True)
cfg.update(checkpoint=history[epoch], resume_epoch=epoch,
           resume_history=[{"epoch": n, "path": history[n]} for n in sorted(history)],
           output_dir=str(out / "outputs"))
config = out / "config.yaml"
config.write_text(yaml.safe_dump(cfg, sort_keys=False))
(out / "resume_manifest.json").write_text(json.dumps({
    "source": history[epoch], "completed_epochs": epoch,
    "total_epochs": cfg["train"]["num_epoch"], "optimizer_restored": True,
    "exact_uninterrupted_trajectory": False,
    "reason": "Legacy checkpoints do not contain random-generator state."
}, indent=2))
print(f"Resume epoch {epoch} -> {cfg['train']['num_epoch']}; logs: {out}", flush=True)
with (out / "console.log").open("w", encoding="utf-8") as log:
    proc = subprocess.Popen(
        [sys.executable, "-u", str(ROOT / "repos/ULTRA/script/run.py"),
         "-c", str(config), "--seed", str(a.seed)],
        cwd=out, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace")
    try:
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        code = proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        proc.wait()
        raise
raise SystemExit(code)
