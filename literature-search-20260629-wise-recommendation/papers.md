# Literature Search: WISE Recommendation Papers

Date: 2026-06-29
Search purpose: identify WISE papers that can be cited naturally in LoCoRec's Related Work
Target venue/family: WISE / Springer LNCS
Source-quality policy: applied; official Springer proceedings were used as primary sources

## Summary

- Closest-work clusters: sequential data augmentation, collaborative-signal augmentation, semantic/contrastive representation learning.
- Recommended citations for the manuscript: NCL4Rec, CSA4Rec, and SDARec.
- Positioning: these methods augment sequences, interaction graphs, or datasets; LoCoRec calibrates the item-level parameter-sharing structure induced by hard Semantic IDs.
- Citation caution: WISE affiliation alone is not sufficient reason to cite a paper; SACORec, CAN, MDAP, and TP-CDSR are useful background but are not close enough to LoCoRec's central mechanism.

## Paper Table

| # | Title | Year | Venue/source | Link | Type | Insight | Completeness | Numeric evidence | Overall | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Semantic-Guided Data Augmentation and Filtering for Sequential Recommendation | 2026 | WISE 2025 | https://doi.org/10.1007/978-981-95-7251-9_4 | pure method | 4 | 4 | 3 | A | Closest WISE paper on semantic augmentation; useful for distinguishing data-level augmentation from SID sharing. |
| 2 | Noise-Augmented Contrastive Learning for Sequential Recommendation | 2023 | WISE 2023 | https://doi.org/10.1007/978-981-99-7254-8_43 | pure method | 3 | 3 | 3 | B | Directly relevant to sequential augmentation under sparsity and noise. |
| 3 | CSA4Rec: Collaborative Signals Augmentation Model Based on GCN for Recommendation | 2025 | WISE 2024 | https://doi.org/10.1007/978-981-96-0570-5_8 | pure method | 3 | 3 | 3 | B | Relevant to controlled collaborative-signal augmentation and residual-style reuse. |
| 4 | Semantic Similarity-Based Graph Contrastive Learning for Recommender System | 2025 | WISE 2024 | https://doi.org/10.1007/978-981-96-0570-5_2 | pure method | 3 | 3 | 3 | B | Useful supporting work on semantic similarity and graph contrastive learning, but not sequential. |
| 5 | Enhancing Multi-behavior Sequential Recommendation via Transformer-Based Cross-Layer Contrastive Learning | 2026 | WISE 2025 | https://doi.org/10.1007/978-981-95-7251-9_2 | pure method | 3 | 3 | 4 | B | Sequential and contrastive, but focused on multi-behavior interactions rather than item sharing. |
| 6 | Self-attention Convolutional Neural Network for Sequential Recommendation | 2023 | WISE 2023 | https://doi.org/10.1007/978-981-99-7254-8_44 | pure method | 2 | 3 | 3 | C | General sequential encoder background; SASRec and BERT4Rec already provide stronger anchors. |
| 7 | Capturing Multi-granularity Interests with Capsule Attentive Network for Sequential Recommendation | 2022 | WISE 2021 | https://doi.org/10.1007/978-3-030-91560-5_11 | pure method | 3 | 3 | 3 | C | Multi-interest user modeling is peripheral to LoCoRec's item representation problem. |
| 8 | Cross-Domain Sequential Recommendation with Temporal Encoding and Projection-Based Learning | 2025 | WISE 2024 | https://doi.org/10.1007/978-981-96-0570-5_6 | pure method | 3 | 3 | 3 | C | Relevant to transfer and negative transfer, but its cross-domain setting differs from LoCoRec. |

## Clusters

### Sequential and semantic augmentation

- Representative papers: NCL4Rec and SDARec.
- What they solve: enrich or denoise training sequences, with SDARec explicitly using item semantics and LLM-based filtering.
- Remaining distinction: they do not modify the item-parameter sharing relations induced by hard Semantic IDs.

### Collaborative and graph augmentation

- Representative papers: CSA4Rec and SSGCL.
- What they solve: augment higher-order collaborative signals or learn semantic graph views.
- Remaining distinction: their sharing is mediated by interaction graphs rather than multi-level SID overlap.

### Sequential encoder and transfer models

- Representative papers: MST-CCL, SACORec, CAN, and TP-CDSR.
- Role: optional background only; they should not be added solely because they appeared at WISE.

## Citation And Positioning Cautions

- Cite NCL4Rec, CSA4Rec, and SDARec as augmentation-based approaches, not as Semantic-ID methods.
- Use the official proceedings year in the BibTeX record. WISE 2024 chapters are cited by Springer as 2025, and WISE 2025 chapters are cited as 2026.
- Do not imply that these papers address hard-SID quantization mismatch or token collisions.
