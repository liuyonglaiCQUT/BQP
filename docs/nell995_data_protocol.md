# NELL-995 data protocol and target population

## Why this dataset

NELL-995 adds a graph with 75,492 entities and 200 base relations, substantially larger and sparser than FB15k-237 and WN18RR. The published benchmark commonly reports 149,678 train, 543 validation, and 3,992 test triples. See the original NELL-995 reasoning benchmark, [DeepPath (Xiong et al., EMNLP 2017)](https://aclanthology.org/D17-1060/), and a published dataset-statistics table in [Learning Attention-based Embeddings for Relation Prediction in Knowledge Graphs (Wang et al., ACL 2019)](https://aclanthology.org/P19-1466/).

The local directory is not a clean copy of that standard split. Its `raw.kb` contains the complete 154,213-fact graph; `train.json` contains that complete graph plus explicit reciprocal relations; and all 1,034 facts in the local `valid.json` also occur in the local 3,710-fact `test.json`. Treating those files as ordinary disjoint KGC splits would leak evaluation facts into training and checkpoint selection.

## Derived split

`scripts/prepare_nell995_bqp.py` leaves the source files untouched and creates `data/NELL995/` from `raw.kb` with a new, relation-stratified split:

| Split | Triples |
|---|---:|
| Train | 149,678 |
| Validation | 543 |
| Test | 3,992 |

The counts match the commonly reported split sizes, but the selected triples do **not** claim to reproduce the canonical benchmark assignment. The split uses seed `20261018`. For every entity, it protects one randomly selected incident fact in train; validation and test are sampled from the remaining candidates, with relation-proportional allocation capped by each relation's candidate count. All 200 relations appear in test and 196 appear in validation. The split files and their SHA-256 hashes are recorded in `data/NELL995/dataset_manifest.json`.

This constraint matters for transductive KGC: an unconstrained random split can place the only occurrence of an entity in validation or test, leaving that entity embedding untrained. The independent audit verifies that all entities occur in train, the splits are pairwise disjoint, relation coverage and split sizes match the manifest, and file hashes are intact. The entity-cover rule changes which facts can be held out, so all results are conditional on this explicitly derived split.

The added dataset combines a large entity vocabulary and sparse training graph with many relation types. Counting head/tail incidences in the training split, FB15k-237 has 14,505 represented entities, 237 relations, mean degree 37.41 and median degree 22; WN18RR has 40,559 represented entities, 11 relations, mean degree 4.28 and median degree 3; NELL-995 has all 75,492 entities represented, 200 relations, mean degree 3.96 and median degree 1. This is descriptive characterization; it is not model input or evidence that held-out facts represent all possible NELL facts.

This is a transductive filtered-ranking setup. The entity candidate set contains all 75,492 entities. All 200 base relations occur in test and 196 occur in validation, given the entity-coverage constraint and available candidates. The repository adds reciprocal queries at load time. Validation checkpoint selection uses `all_known` filtering, and final test ranking filters other known facts from train, validation, and test. Ties use average rank.

## Target population and interpretation

The finite target population for the third-dataset analysis is the 3,992 positive source triples in this particular entity-covered derived test split, under its constrained relation-stratified allocation. Each source triple contributes a paired tail and head query. We condition on the fixed raw graph, derived split, and checkpoint.

The bootstrap experiment treats that finite test set as an empirical distribution and asks how intervals behave under hypothetical repeated source-triple samples from it. The full-test MRR is an exact descriptive value for the fixed test set; its bootstrap interval is **not** uncertainty about that known finite-set value. It is not an estimate over all NELL facts, all possible NELL splits, future real-world facts, other knowledge graphs, or training-seed variation. Those broader targets would require additional sampling assumptions and experiments.

This distinction also applies to FB15k-237 and WN18RR. Unless a paper defines a credible sampling design beyond a fixed benchmark split, BQP's intervals should be reported as conditional resampling diagnostics rather than universal uncertainty guarantees.

## Training and evaluation matrix

The NELL extension is limited to the core ComplEx (rank 200) and DistMult (rank 400) models, each with seeds 7, 11, 19, 23, and 31. Both use reciprocal-query negative-sampling logistic loss, Adam, 100 maximum epochs, batch size 1,024, 128 negatives per positive, learning rate 0.001, validation every five epochs, patience eight, and N3 coefficient `1e-4`. Entity and relation embeddings use normal initialization with scale 0.05. Checkpoint selection maximizes all-known filtered validation MRR. These settings are frozen in code before the corrected split's test ranks are exported, and each run's `config.json` and `manifest.json` record the actual settings, selected epoch, hashes, and validation trajectory. The test split is not used for checkpoint selection.

The older FB15k-237/WN18RR ComplEx/DistMult checkpoints retain their model and evaluation settings, but their historical optimizer, loss, and sampling configuration is not fully available locally. A separate local-retraining recipe is being run under `checkpoints/bqp_local_reproduction/` and documented in `local_reproduction_protocol.md`. It makes the new models reproducible but does not reconstruct the historical configurations or weights. Cross-dataset comparisons involving the historical core checkpoints remain descriptive and cannot isolate graph structure from training implementation.

Commands:

```powershell
python scripts/prepare_nell995_bqp.py
python scripts/audit_nell995_split.py
python scripts/train_bqp_additional_architectures.py --dataset NELL995 --model complex --model distmult --device cuda
python scripts/export_bqp_test_ranks.py --dataset NELL995 --device cuda
```

