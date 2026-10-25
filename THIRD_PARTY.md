# Third-party source and provenance

This repository contains adapters around three external model families. The
upstream research code remains attributable to its original authors.

## CompoundE

- Upstream: <https://github.com/hughxiouge/CompoundE>
- Pinned commit: `ed370d44ad39a43941b42b75112e5e1e45311cb1`
- Location: `model_extensions/repos/CompoundE/`
- License: MIT; the upstream `LICENSE` file is preserved.
- Local integration: the FB15k-237/WN18RR experiment uses
  `model_extensions/adapters/compounde.py` and
  `model_extensions/scripts/run_compounde.py`. The untouched upstream source
  is included for provenance and reference.

## ULTRA

- Upstream: <https://github.com/DeepGraphLearning/ULTRA>
- Pinned commit: `427966ad8ed60420eef034063d44f3153addff90`
- Location: `model_extensions/repos/ULTRA/`
- License: MIT; the upstream `LICENSE` file is preserved.
- Excluded: upstream pretrained checkpoints under `ckpts/`.
- Local changes: single-GPU resume support in `script/run.py` and a CUDA stream
  API compatibility update in `ultra/rspmm/source/rspmm.cu`. The exact diff is
  recorded in `model_extensions/patches/ULTRA.patch`.

## SimKGC

- Upstream: <https://github.com/intfloat/SimKGC>
- Pinned commit: `97cc43e488f19ca5b0f6fbf60ffefd2ee56c0693`
- Local patch: `model_extensions/patches/SimKGC.patch`
- Fetch command: `bash model_extensions/scripts/setup_third_party.sh`

The pinned SimKGC revision does not contain a license file. Its source is
therefore not copied into this release. The setup script asks Git to obtain the
source directly from the official repository and applies the local patch in
the user's working tree. Before vendoring or redistributing SimKGC itself,
obtain permission or confirm an applicable upstream license.

## Data and pretrained models

No third-party dataset or pretrained checkpoint is included. Dataset and
weight licenses/terms remain those of their respective providers. Users are
responsible for obtaining them from the official sources.
