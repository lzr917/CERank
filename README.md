# CERank

Code for *Complementing Cross-Domain Preference Modeling with Collaborative
Evidence for Candidate Ranking*. CERank uses target-domain collaborative
evidence to refine candidate representations for a frozen recommender.

## Run

Use Python 3.9. Install the PyTorch 2.0.1 wheel for your CPU or CUDA environment,
then install the remaining dependencies with `python -m pip install -r requirements.txt`.

Place processed data in `data/<dataset>/`. The loader expects `num.txt` with
the A and B item counts, plus `train_A.txt`, `train_B.txt`,
`train_overlap.txt`, `valid.txt`, and `test.txt`. Each sequence row uses
`session_id<TAB>item0,item1,...` with zero-based item IDs and the B IDs offset
by the A item count. Provide your own data before running the commands below.

```bash
python cerank/scripts/train_backbone.py --backbone sasrec --dataset food-kitchen-5core --direction B2A --seed 42 --device cuda:0
python cerank/scripts/train_adapter.py --backbone sasrec --dataset food-kitchen-5core --direction B2A --seed 42 --device cuda:0 --disable_legacy_cache
```

The second command loads the backbone checkpoint produced by the first and
reports validation and test metrics. Results are written under `cerank/results/`.
For C2DSR, use `--backbone c2dsr` and pass `--official_root` with the path to
your upstream C2DSR checkout on both commands. The path must contain
`C2DSR_src/model/`.

The adapter and multi-hop evidence implementation are in
[`plugin.py`](cerank/src/cerank/plugin.py) and
[`structural.py`](cerank/src/cerank/structural.py), respectively.

