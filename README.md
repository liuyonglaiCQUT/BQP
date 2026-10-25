# BQP-KGC

Code for **Bidirectional Query Pseudoreplication (BQP)** in knowledge graph
completion evaluation. A test triple produces a tail query and a head query.
The two reciprocal ranks share a source fact and can therefore be correlated.
This repository compares query-independent inference with source-triple-level
inference while leaving the conventional MRR point estimate unchanged.

This is the release-oriented source tree for the paper:

> *One Fact Is Not Two Independent Samples: Bidirectional Query
> Pseudoreplication in Knowledge Graph Completion Evaluation*

The repository intentionally contains **no benchmark data, trained
checkpoints, pretrained language-model weights, rank exports, or generated
results**. Those files are produced or downloaded by the steps below and are
ignored by Git.

## What is included

- paired head/tail statistics, cluster standard errors, bootstrap confidence
  intervals, within-relation re-pairing, multiple-testing correction, and
  coverage simulations;
- implementations and training/evaluation code for ComplEx, DistMult, CP, and
  RotatE;
- the CompoundE adaptation used in the experiments;
- the ULTRA source snapshot and local compatibility changes used in the
  experiments, with its upstream MIT license;
- wrappers, patches, and validators for SimKGC at a pinned upstream commit;
- rank-export audits, experiment drivers, and figure-generation scripts;
- frozen protocol documents under `docs/`.

The paper's frozen checkpoint matrix is:

| Model family | Datasets | Seeds/checkpoints | Runs |
|---|---|---:|---:|
| ComplEx, DistMult | FB15k-237, WN18RR, derived NELL-995 | 7, 11, 19, 23, 31 | 30 |
| CP, RotatE | FB15k-237, WN18RR | 7, 11, 19, 23, 31 | 20 |
| CompoundE | FB15k-237, WN18RR | 7, 11, 19, 23, 31 | 10 |
| SimKGC, ULTRA | FB15k-237, WN18RR | fixed checkpoint, seed 7 | 4 |
| **Total** |  |  | **64** |

The five-seed algorithm-level comparisons in the paper use ComplEx, DistMult,
CP, and RotatE. SimKGC and ULTRA are fixed-checkpoint architecture
replications, not five-seed algorithm-level estimates.

## Repository layout

```text
BQP-KGC/
├── bqp/                         # data loader, KGE models, metrics, BQP statistics
├── scripts/                     # core training, rank export, analyses, audits, plots
├── docs/                        # frozen statistical and evaluation protocols
├── model_extensions/
│   ├── adapters/                # CompoundE and shared BQP adapters
│   ├── patches/                 # recorded changes to upstream sources
│   ├── repos/                   # vendored MIT sources; SimKGC is fetched on demand
│   └── scripts/                 # setup, training, resume, evaluation, validation
├── requirements.txt             # core analysis and four-model environment
├── requirements-model-extensions.txt
├── THIRD_PARTY.md
└── .gitignore
```

Generated directories such as `data/`, `checkpoints/`, `results_bqp/`,
`model_extensions/runs/`, and `model_extensions/results/` are not part of the
source release.

## 1. Core environment

The core code was developed with Python 3.10. Statistical analysis can run on
CPU; training and full-entity filtered ranking are substantially faster on a
CUDA GPU.

```bash
cd BQP-KGC
python -m venv .venv
source .venv/bin/activate       # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/smoke_bqp_statistics.py
```

The released requirements intentionally specify compatible ranges for the
core analysis. For the extension experiments, use the separately pinned
environment below.

## 2. Model-extension environment

The extension runs were executed on Ubuntu with one NVIDIA RTX 4090, Python
3.10, PyTorch 2.1.0+cu118, and CUDA Toolkit 11.8. A driver capable of running
CUDA 11.8, Conda, a compatible C++ compiler, and Bash are required. The setup
does not use any weights stored in this repository.

First fetch the pinned SimKGC source and apply the recorded local patch:

```bash
bash model_extensions/scripts/setup_third_party.sh
```

Download the official ULTRA `ultra_3g.pth` checkpoint and place it at:

```text
model_extensions/repos/ULTRA/ckpts/ultra_3g.pth
```

Then create the isolated extension environment and run preflight tests. The
preflight includes a strict ULTRA weight-loading check, so the checkpoint must
already be present:

```bash
bash model_extensions/scripts/setup_server.sh
bash model_extensions/scripts/preflight.sh
```

`setup_third_party.sh` checks out SimKGC commit
`97cc43e488f19ca5b0f6fbf60ffefd2ee56c0693` and applies
`model_extensions/patches/SimKGC.patch`. CompoundE and ULTRA are vendored at
the commits recorded in `THIRD_PARTY.md`. SimKGC downloads
`bert-base-uncased` through Hugging Face unless
`SIMKGC_PRETRAINED_MODEL` points to a local model directory.

## 3. Data preparation

No dataset is redistributed. The core loaders expect tab-separated triples
in the following layout:

```text
data/
├── FB15k237/{train,valid,test}.txt
├── WN18RR/{train,valid,test}.txt
└── NELL-995/
    ├── raw.kb
    ├── train.json
    ├── valid.json
    └── test.json
```

Each text row must contain `head relation tail`. FB15k-237 and WN18RR can be
obtained from their standard public distributions. The pinned SimKGC checkout
also contains the FB15k-237/WN18RR source files and text metadata required by
SimKGC; copy the three triple files into the core layout without modifying row
content:

```bash
mkdir -p data/FB15k237 data/WN18RR
cp model_extensions/repos/SimKGC/data/FB15k237/{train,valid,test}.txt data/FB15k237/
cp model_extensions/repos/SimKGC/data/WN18RR/{train,valid,test}.txt data/WN18RR/
```

The paper does not treat the locally available NELL-995 JSON files as a clean
benchmark split. It deterministically derives an entity-covered,
relation-stratified split from `data/NELL-995/raw.kb`:

```bash
python scripts/prepare_nell995_bqp.py
python scripts/audit_nell995_split.py
```

This creates `data/NELL995/` with 149,678/543/3,992
train/validation/test triples. It is a documented derived split, not a claim
to reproduce the canonical NELL-995 fact assignment. See
`docs/nell995_data_protocol.md`.

For CompoundE and SimKGC, the extension wrappers read data below
`model_extensions/repos/SimKGC/data/`. `prepare_simkgc.py` converts the raw
files to the official SimKGC JSON format. ULTRA downloads/processes data under
`model_extensions/data/ULTRA/` through its official dataset loader.

## 4. Reproducing the four-model core matrix

Train validation-selected checkpoints for all five seeds. The defaults are
the frozen presets in `scripts/train_bqp_additional_architectures.py`.

```bash
python scripts/train_bqp_additional_architectures.py \
  --dataset FB15k237 --dataset WN18RR \
  --model complex --model distmult --model cp --model rotate \
  --device cuda

python scripts/train_bqp_additional_architectures.py \
  --dataset NELL995 \
  --model complex --model distmult \
  --device cuda
```

Checkpoints are selected only by filtered validation MRR. Export paired test
ranks, then audit identity, direction order, filtering, and metric recovery:

```bash
python scripts/export_bqp_test_ranks.py --device cuda
python scripts/audit_bqp_test_ranks.py
python scripts/check_bqp_formal_matrix_status.py
```

The main rank export for each run is:

```text
results_bqp/test_query_ranks/<dataset>/<model>/seed_<seed>/test_query_ranks.npz
```

It stores the source-triple ID, head/tail direction, raw triple identity,
filtered rank, and reciprocal rank. Do not remove the source-triple ID before
running BQP inference.

Run the statistical analyses:

```bash
python scripts/run_bqp_core_experiment.py
python scripts/audit_bqp_core_experiment.py

python scripts/run_bqp_coverage_experiment.py
python scripts/audit_bqp_coverage_experiment.py

python scripts/run_bqp_architecture_extension.py --device cuda
python scripts/audit_bqp_architecture_extension.py

python scripts/run_bqp_near_tie_experiment.py
python scripts/audit_bqp_near_tie_experiment.py
```

Default formal settings include 50,000 bootstrap replicates and 5,000
within-relation permutations in the core experiment, 3,000 coverage
replicates, and up to 200,000 confirmation replicates for decision-boundary
cases. Use reduced counts only for debugging and do not report them as formal
paper results.

## 5. CompoundE replication

The local adapter implements the paper's FB15k-237 and WN18RR variants and
records all training/selection metadata. Run all five seeds:

```bash
for seed in 7 11 19 23 31; do
  for dataset in FB15k237 WN18RR; do
    bash model_extensions/scripts/run_compounde.sh "$dataset" "$seed"
    bash model_extensions/scripts/validate_compounde_bqp.sh \
      "$dataset" --seed "$seed" --repeats 5000
  done
done
```

Interrupted training can be resumed without overwriting the run:

```bash
bash model_extensions/scripts/run_compounde.sh WN18RR 19 --resume
```

## 6. SimKGC fixed-checkpoint replication

The paper uses one audited checkpoint (seed 7) per dataset for SimKGC:

```bash
for dataset in FB15k237 WN18RR; do
  bash model_extensions/scripts/run_simkgc.sh "$dataset" 7
  bash model_extensions/scripts/validate_simkgc_bqp.sh \
    "$dataset" --seed 7 --repeats 5000
done
```

If at least one complete epoch has been saved, resume with:

```bash
bash model_extensions/scripts/resume_simkgc.sh WN18RR 7
```

The wrapper keeps batch size 1,024 and uses gradient checkpointing rather than
replacing in-batch negatives with gradient accumulation.

## 7. ULTRA fixed-checkpoint replication

Place the official `ultra_3g.pth` as described above, then fine-tune and audit
the validation-selected seed-7 checkpoint:

```bash
for dataset in FB15k237 WN18RR; do
  bash model_extensions/scripts/run_ultra.sh "$dataset" finetune 7
  bash model_extensions/scripts/validate_ultra_bqp.sh \
    "$dataset" --seed 7 --repeats 5000
done
```

The validator records both ULTRA's official pessimistic tie ranks and the
harmonized average-tie ranks used for cross-family BQP summaries.

## 8. Ranking protocols

- Core ComplEx/DistMult/CP/RotatE and CompoundE use train+valid+test filtered
  ranking with average ties.
- SimKGC retains its audited official descending-sort ranking after graph
  reranking.
- ULTRA records its official pessimistic ranking and a harmonized average-tie
  version.

The BQP statistic is applied to each model's audited per-triple rank export.
Do not silently combine results under different tie policies; use the
harmonized rank field when a cross-family comparison requires it.

## 9. Generated outputs and figures

Core outputs are written below `results_bqp/`; extension outputs are written
below `model_extensions/runs/` and `model_extensions/results/`. Important plot
entry points include:

```bash
python scripts/redraw_single_model_coverage.py
python scripts/plot_bqp_appendix_a1.py
python scripts/plot_bqp_appendix_a2.py
python scripts/plot_bqp_appendix_a4.py
python scripts/plot_bqp_appendix_a7.py
python scripts/plot_algorithm_level_ci_tkde.py
```

Every formal analysis should be preceded by the matching audit script. The
audit files and generated manifests are part of a reproducible run, but they
are outputs rather than source and are therefore not bundled here.

## 10. Reproducibility boundary

Because checkpoints are intentionally not distributed, a fresh run reproduces
the documented training and inference procedures rather than the exact bytes
of the authors' trained parameters. GPU kernels, package builds, and stochastic
optimization can cause small numerical differences. Statistical analyses are
deterministically seeded once the paired rank exports are fixed.

The confidence intervals quantify conditional resampling variation over the
fixed benchmark test triples. They do not by themselves quantify uncertainty
over datasets, graph construction, hyperparameter search, or future facts.

## Third-party code and licensing

See `THIRD_PARTY.md`. CompoundE and ULTRA retain their upstream MIT licenses.
SimKGC is fetched from its official repository rather than redistributed
because the pinned upstream revision does not include a license file.

Before publishing this repository, the authors must add an explicit license
for the original BQP code. Without a project license, the repository is
source-available but is not an open-source release in the legal sense.

## Citation

This repository accompanies the following manuscript. Update this entry with
the final bibliographic record after publication:

```bibtex
@unpublished{liu_bqp_kgc,
  title   = {One Fact Is Not Two Independent Samples: Bidirectional Query
             Pseudoreplication in Knowledge Graph Completion Evaluation},
  author  = {Yonglai Liu and Yanjun Lu and Shuhan Nie},
  note    = {Manuscript},
  year    = {2026}
}
```
