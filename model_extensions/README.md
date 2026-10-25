# Model extensions

This directory contains the experiment adapters for CompoundE, SimKGC, and
ULTRA. It is intentionally isolated from the four-model core implementation.

## Setup order

```bash
bash scripts/setup_third_party.sh
bash scripts/setup_server.sh
bash scripts/preflight.sh
```

All generated environments, downloads, caches, datasets, logs, checkpoints,
and BQP outputs remain under this directory and are ignored by Git.

## Main entry points

| Model | Train/evaluate | BQP audit |
|---|---|---|
| CompoundE | `scripts/run_compounde.sh` | `scripts/validate_compounde_bqp.sh` |
| SimKGC | `scripts/run_simkgc.sh` | `scripts/validate_simkgc_bqp.sh` |
| ULTRA | `scripts/run_ultra.sh` | `scripts/validate_ultra_bqp.sh` |

Resume entry points are provided for all three models. A smoke test verifies
code paths only and must not be reported as a benchmark replication.

Formal commands, the frozen checkpoint matrix, and data/weight requirements
are documented in the repository-level `README.md`.
