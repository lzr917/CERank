# CERank

# Minimal Runnable CERank Package

This package is for **paper–code consistency review**, including review in
ChatGPT. Its scope is deliberately **C2DSR + SASRec, Food–Kitchen, B2A,
seed=42**, not all backbones, datasets, seeds or experiments in the paper.
CERank is the current method name. The original internal names
`tcseb_native` / `TCSEBAdapter` are retained so renaming does not accidentally
change the result-generating implementation. Read `CODE_MAP.md` alongside the
paper. `VALIDATION.md` records checks actually executed for this package.

## 1. Install

Tested on Linux with Python 3.9.25, PyTorch 2.0.1+cu118, NumPy 1.26.4,
SciPy 1.13.1 and RecBole 1.2.1. The `>=3.9` declaration in `pyproject.toml`
is a minimum version. Python 3.9.25 is the only version validated for this
package. **GPU is not required for review or smoke tests.** Use Python 3.9
and install the PyTorch 2.0.1 CPU/CUDA wheel suitable for your host, then:

```bash
python -m pip install -r requirements.txt
python -m pip install --no-deps --no-build-isolation -e .
python scripts/validate.py
python -m unittest discover -s tests -v
```

The commands below run from this directory. They use the current Python,
relative package paths, and no original server files or legacy caches.
Use the root `requirements.txt` for this package. The vendored C2DSR README
describes the upstream project's original environment.

## 2. Minimal execution test — NOT a paper experiment

```bash
python scripts/smoke.py
```

This invokes the actual C2DSR and SASRec trainers and the full CERank training
chain on a small set of temporary **synthetic** sequences. The smoke launcher
uses training batches of 4, evaluation batches of 2, and zero data-loader
workers. The `--check_only` path runs one training minibatch per stage.
The model architectures and training objectives are preserved. C2DSR backbone
training uses 300 sampled negatives per positive, SASRec backbone training uses
cross-entropy over the full target item catalog, and CERank adapter training
uses 100 sampled negatives per positive. Validation and test ranking in both
paths use one positive and 999 sampled negatives per record.

The smoke verifies backbone training, freezing, identity initialization,
graph/evidence generation, adapter training, checkpoint I/O and ranking. Its
metrics are marked `smoke_only=true` and must never be reported as Food–Kitchen
results. Synthetic checkpoints and caches are temporary and are not shipped.

## 3. Real checkpoint evaluation — reproduces one stored paper setting

```bash
python scripts/evaluate.py --backbone sasrec --device cpu
```

The package includes **six processed Food–Kitchen data files** (five split
files and the item counts in `num.txt`), original seed-42 SASRec weights,
trained CERank weights, and their calibration buffers.
This command reconstructs the actual train-only affinity and test support
using the original code, evaluates the **300 real test records** with one
positive +999 negatives, and checks all four displayed metrics against the
stored historical result within 2e-6. No training is performed.

Expected NDCG@10 is about **0.328264 → 0.337789** (Base → CERank).
These are one seed's results, not the paper's three-seed mean. Results are
written under `outputs/`. Both backbone/adapter weights are losslessly exported:
all parameters and buffers are identical to the original tensors; unused
checkpoint metadata containing server paths is omitted. No precision or
architecture conversion was applied.

## 4. Formal training generators — optional, NOT run during packaging

```bash
# Train CERank using the bundled exact frozen SASRec seed-42 backbone.
python scripts/run.py --backbone sasrec --stage adapter --device cuda:0

# C2DSR's large historical checkpoint is intentionally omitted.
# This genuine training pipeline generates a base, then freezes it for CERank.
python scripts/run.py --backbone c2dsr --stage both --device cuda:0

# Optional actual mechanism controls on the same frozen SASRec backbone.
python scripts/run.py --backbone sasrec --variant adapter_only --device cuda:0
python scripts/run.py --backbone sasrec --variant shuffled_evidence --device cuda:0

# Evaluate a generated run; change backbone/variant to match the training.
python scripts/evaluate.py --backbone c2dsr --run-root outputs --device cpu
python scripts/evaluate.py --backbone sasrec --variant adapter_only --run-root outputs --device cpu

# Print commands without training or filesystem writes.
python scripts/run.py --backbone c2dsr --stage both --dry-run
```

Formal training keeps the original 100/10 backbone budget, 50/10 adapter
budget, batch128/eval256, sequence length50, losses and validation selection.
SASRec uses native full-target-catalog backbone CE; C2DSR uses its original
protocol-aligned supervised/MI objective. Formal adapter training may take
time; it is not required for review. No large experiment was rerun to build
this ZIP. Fresh training across devices is not promised to be bitwise equal
to the historical GPU run. A file lock prevents overlapping writes to the
same backbone queue. Completed runs are checked before skipping; `--rerun`
explicitly permits replacing outputs, never the bundled pretrained artifacts.

`adapter_only` uses the actual historical `neural_only` switch combination:
eta and coverage are zeroed and trust is bypassed; neural/history features and
the residual head remain. `shuffled_evidence` uses the historical deterministic
record/candidate evidence shuffling. Neither control is invented for this ZIP.

## Included resources and omissions

Included: actual numerical code, two bridges, three unchanged configs, six
processed Food–Kitchen data files (already indexed), a small SASRec
checkpoint and CERank checkpoint, calibration statistics/buffers, upstream C2DSR
model/GNN plus its license, a four-row reference table, tests and instructions.

Generated rather than included: affinity matrix, multi-hop evidence, training
candidate caches and C2DSR checkpoints. Their real executable generators are
in the shipped original trainers/structural code. Evaluation reuses the saved
training calibration buffers; it never estimates calibration on test data.

Not included: Movie/Toys, seeds43/44, other backbones, full experiment tables,
large historical C2DSR checkpoint (~153 MiB), hyperparameter queues, figures,
drafts, private handoffs, semantic/LLM features, full logs, Git history or old
cache trees. The package therefore cannot independently verify all 21 main
settings, three-seed significance claims, or all manuscript figures. It can
review the complete CERank mechanism and execute both representative paths.

The upstream C2DSR files retain their original MIT license. See
`THIRD_PARTY.md` for dependency/data attribution and the authors' license status.
Code review scope is not a new blanket redistribution license for the data.
