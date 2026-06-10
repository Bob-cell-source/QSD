# LC-SoftCRSID Reviewer Evidence Experiments

This package addresses reviewer concerns without changing the formal main method or rerunning multiple random seeds.

## Evidence Included

1. **Multi-slot neighborhood validity**: measures text cosine similarity, title overlap, brand consistency, category consistency, and train-only behavioral co-occurrence for item pairs sharing 1, 2, 3, or 4 SID slots.
2. **Scalability**: reports the naive comparison count, slot-inverted-index traversal count, two-slot combination-index traversal count, actual Soft SID construction time, and tensor memory.
3. **Alternative softening**: compares the proposed two-slot local-consistent neighborhood against single-slot global smoothing and text-embedding kNN smoothing.
4. **Reliability calibration**: reports correlations between reliability, item frequency, log-frequency, and Soft SID entropy. When a checkpoint is available, test performance is reported by reliability quantile.
5. **Compact sensitivity**: varies only the three structural parameters: candidate count $M$, overlap threshold $\delta$, and allocation factor $\tau$.

The formal defaults remain unchanged:

```text
M=4, delta=2, min_support=0.05, eta=2,
hard_prior=1, H=50, tau=20
```

## Run Office

```bash
DATASET_DIR=runs/office \
SEMANTIC_IDS=runs/office/semantic_ids_rq.json \
EMBEDDINGS=runs/office/item_text_embeddings.npy \
EMBEDDING_ITEM_IDS=runs/office/embedding_item_ids.json \
BATCH_SIZE=256 \
EVAL_BATCH_EVAL_SIZE=256 \
bash scripts/run_lcsoft_reviewer_evidence.sh
```

For an 8 GB GPU, keep both batch sizes at 128 or 256.

## Run Diagnostics Only

```bash
DATASET_DIR=runs/office \
RUN_TRAINING=0 \
bash scripts/run_lcsoft_reviewer_evidence.sh
```

## Run Another Dataset

```bash
DATASET_DIR=runs/beauty \
BATCH_SIZE=1024 \
EVAL_BATCH_EVAL_SIZE=1024 \
bash scripts/run_lcsoft_reviewer_evidence.sh
```

The dataset directory must contain:

```text
sequences.json
stats.json
item_meta.json
semantic_ids_rq.json
item_text_embeddings.npy
embedding_item_ids.json
```

If text embeddings are unavailable, the script skips text-based diagnostics and the text-kNN baseline while retaining SID diagnostics and sensitivity experiments.

## Outputs

All artifacts are isolated under:

```text
runs/<dataset>/reviewer_evidence_20260610/
```

Important files:

```text
diagnostics/neighbor_evidence.json
diagnostics/neighbor_evidence.csv
diagnostics/reliability_structure.json
diagnostics/reliability_with_performance.csv
training/summary.csv
training/*/soft_sid_preprocess.json
```

`FORCE=0` is the default, so existing results are never overwritten. Set `FORCE=1` only when an explicit rerun is required.

## Interpretation

- Text/category/behavior consistency should increase with overlap slots. Otherwise, the multi-slot neighborhood assumption is not supported.
- LC-SoftSID should outperform both single-slot smoothing and text-kNN smoothing. Otherwise, the gain cannot be attributed specifically to local multi-slot consistency.
- Reliability should not be almost perfectly correlated with frequency. Moderate correlation is acceptable, but reliability must provide additional information.
- Performance should generally improve across reliability bins. A non-monotonic result should be discussed as a calibration limitation.
- Sensitivity results should show a stable region around the default configuration rather than a single sharp optimum.

The RQ-KMeans files currently store hard codes but not residual codebook centers. Therefore, a true nearest-RQ-centroid probability baseline requires rebuilding Semantic IDs with saved per-level residual centroids. It is intentionally not mislabeled or approximated in this package.
