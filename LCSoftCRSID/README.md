# LC-SoftCRSID

This directory contains a clean implementation of the paper method, separated
from the historical QSDRec code and exploratory GRU experiments.

## Method pipeline

1. Encode item metadata and construct hard Semantic IDs with residual K-Means.
2. Build a slot-aware local neighborhood from hard-SID overlap.
3. Convert each hard SID into a local-consistent weighted Top-M Soft SID.
4. Construct each item from a semantic basis, shared semantic residual, and
   private item residual.
5. Calibrate shared/private residual allocation with item frequency and Soft-SID
   reliability.
6. Feed the resulting item representations to a causal Transformer and optimize
   sampled next-item cross-entropy.

## Directory

```text
LCSoftCRSID/
├── build_semantic_ids.py       # Hard SID construction entry point
├── train.py                    # Training entry point
├── scripts/run_main.sh         # Main paper configuration
└── lcsoftcrsid/
    ├── data.py                 # Leave-one-out data and negative sampling
    ├── io.py                   # JSON helpers
    ├── model.py                # LC-SoftCRSID and causal Transformer
    ├── semantic_id.py          # Text encoding and residual K-Means
    ├── soft_sid.py             # Local-consistent Soft SID construction
    └── trainer.py              # Training and full-ranking evaluation
```

## Expected dataset files

```text
runs/<dataset>/
├── sequences.json
├── stats.json
├── item_meta.json
└── semantic_ids_rq.json
```

`sequences.json` contains rows with an `items` field. The last two interactions
are reserved for validation and testing. Item IDs must be consecutive integers
from 1 to `num_items`.

## Build hard Semantic IDs

```bash
python LCSoftCRSID/build_semantic_ids.py \
  --item-meta runs/beauty/item_meta.json \
  --output runs/beauty/semantic_ids_rq.json \
  --encoder-model sentence-transformers/all-MiniLM-L6-v2 \
  --codebook-sizes 64,128,256,512 \
  --device cuda
```

## Train the main method

```bash
DATASET_DIR=runs/beauty bash LCSoftCRSID/scripts/run_main.sh
```

Or call the entry point directly:

```bash
python LCSoftCRSID/train.py \
  --dataset-dir runs/beauty \
  --semantic-ids runs/beauty/semantic_ids_rq.json \
  --output-dir runs/beauty/lcsoftcrsid_clean/main \
  --device cuda
```

The default arguments reproduce the current Transformer main method:

```text
dim=128, layers=2, heads=2, dropout=0.2
tau=20, residual_scale=1.0, frequency_transform=raw
top_m=4, overlap_slots=2, min_support=0.05
support_eta=2.0, hard_prior=1.0, reliability_floor=0.1
max_neighbors=50, random_negatives=100
```

The clean implementation intentionally excludes QSDRec, GRU transfer probes,
behavior-neighbor experiments, local-lift experiments, and unused score-level
semantic branches.
