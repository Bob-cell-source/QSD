# Large-Dataset Incremental Validation

This experiment package validates Beauty, Sports, and Toys/Games without rerunning completed baselines or module ablations.

## Experiments Added

Per dataset, the script runs only:

1. Multi-slot neighbor quality, preprocessing complexity, and reliability calibration diagnostics.
2. Single-slot Soft SID as a generic global-smoothing baseline.
3. Text-kNN Soft SID as an external semantic-neighbor baseline when text embeddings exist.
4. The missing structural sensitivity points: $\delta=3$, $\tau=5$, and $\tau=80$.
5. One shared fixed control and five learnable variants:
   - unconstrained candidate attention;
   - prior-guided candidate attention;
   - prior-guided attention with KL regularization;
   - learnable monotonic alpha;
   - learnable alpha with prior-guided attention.

The script does not rerun SASRec, QSDRec, Hard CRSID, the formal LC-SoftCRSID main experiment, $M=8$, no-pruning, $\eta=1$, module removal, behavior neighbors, local lift, or multiple random seeds.

## Run All Three Datasets

```bash
BATCH_SIZE=512 \
EVAL_CHUNK_SIZE=1024 \
bash scripts/run_lcsoft_large_dataset_validation.sh
```

For a smaller GPU:

```bash
BATCH_SIZE=128 \
EVAL_CHUNK_SIZE=256 \
bash scripts/run_lcsoft_large_dataset_validation.sh
```

## Run One Dataset

```bash
DATASETS="runs/sports" \
BATCH_SIZE=512 \
EVAL_CHUNK_SIZE=1024 \
bash scripts/run_lcsoft_large_dataset_validation.sh
```

## Run Selected Sections

Diagnostics only:

```bash
RUN_REVIEWER_TRAINING=0 \
RUN_LEARNABLE_PROBE=0 \
bash scripts/run_lcsoft_large_dataset_validation.sh
```

Learnable attention and alpha only:

```bash
RUN_DIAGNOSTICS=0 \
RUN_REVIEWER_TRAINING=0 \
bash scripts/run_lcsoft_large_dataset_validation.sh
```

Reviewer softening and sensitivity only:

```bash
RUN_DIAGNOSTICS=0 \
RUN_LEARNABLE_PROBE=0 \
bash scripts/run_lcsoft_large_dataset_validation.sh
```

## Required Files

Each dataset directory requires:

```text
sequences.json
stats.json
item_meta.json
semantic_ids_rq.json
```

Text diagnostics and the text-kNN baseline additionally require:

```text
item_text_embeddings.npy
embedding_item_ids.json
```

Without text embeddings, those two components are skipped automatically.

For datasets with more than 10,000 items, install FAISS before running the text-kNN baseline:

```bash
pip install faiss-cpu
```

Without FAISS, large text-kNN training is skipped to avoid an accidental slow exact $N^2$ search. To explicitly permit the chunked exact implementation, set `ALLOW_EXACT_TEXT_KNN=1`.

## Existing Main Result

The script does not retrain the main method. It looks for:

```text
runs/<dataset>/lc_soft_required_ablation/20_lc_soft_full/test_metrics.json
```

or:

```text
runs/<dataset>/lc_soft_crsid_ablation/20_lc_soft_crsid_full_m4_eta2/test_metrics.json
```

and inserts the existing result into the final summary.

## Output

```text
runs/<dataset>/lcsoft_incremental_validation_20260610/
  diagnostics/
  reviewer_training/
  learnable_probe/
  summary.csv
```

Existing completed experiments are skipped unless `FORCE=1` is explicitly set.
