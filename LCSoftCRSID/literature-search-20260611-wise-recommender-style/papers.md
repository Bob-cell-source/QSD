# Literature Search: Recent Recommender-System Papers at WISE

Date: 2026-06-11
Search purpose: identify recent WISE recommender-system papers and extract venue-specific writing and organization patterns for LC-SOFT CRSID.
Target venue: International Conference on Web Information Systems Engineering (WISE)
Source-quality policy: official Springer LNCS pages, DBLP proceedings, and author-released paper pages were prioritized.

## Summary

- WISE 2024 Part III contains a dedicated Recommendation Systems section in LNCS 15438.
- The section includes work on sequential, cross-domain, multi-behavior, multimodal, graph, POI, and news recommendation.
- The closest structural exemplars for LC-SOFT CRSID are the WISE 2024 cross-domain sequential recommendation paper, the heterogeneous Transformer sequence recommendation paper, and MDAP.
- Recent WISE recommendation papers favor a compact challenge-to-module story, an explicit problem-definition subsection, an overview figure, module-by-module equations, and experiments ordered as settings, overall comparison, ablation, and analysis.
- Springer LNCS controls typesetting; annual WISE submission length and anonymity rules must be verified from the target-year Call for Papers.

## Paper Table

| # | Title | Year | Venue/source | Link | Type | Insight | Completeness | Numeric evidence | Overall | Relevance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Cross-Domain Sequential Recommendation with Temporal Encoding and Projection-Based Learning | 2024 | WISE 2024, LNCS 15438 | [Springer](https://link.springer.com/chapter/10.1007/978-981-96-0570-5_6) | pure method | 4 | 4 | 4 | A | Closest sequential-recommendation structure; abstract maps two challenges to two modules. |
| 2 | The Research of Sequence Recommendation Method Based on Heterogeneous Enhanced Transformer with Multi-behavior Data | 2024 | WISE 2024, LNCS 15438 | [Springer](https://link.springer.com/chapter/10.1007/978-981-96-0570-5_11) | pure method | 3 | 4 | 4 | A | Transformer-based sequence recommendation with multiple ordered modules. |
| 3 | MDAP: A Multi-view Disentangled and Adaptive Preference Learning Framework for Cross-Domain Recommendation | 2024 | WISE 2024, LNCS 15438 | [full text](https://ar5iv.org/abs/2410.05877) | pure method | 4 | 5 | 4 | A | Best accessible exemplar for full section organization, formulas, algorithm, experiments, and ablation. |
| 4 | Causal Behavior Pattern Inference for News Recommendation Through Multi-interest Matching | 2024 | WISE 2024, LNCS 15438 | [Springer](https://link.springer.com/chapter/10.1007/978-981-96-0570-5_13) | pure method | 4 | 4 | 4 | A | Strong problem–insight–mechanism abstract with a concise application motivation. |
| 5 | MIN: Multi-stage Interactive Network for Multimodal Recommendation | 2024 | WISE 2024, LNCS 15438 | [Springer](https://link.springer.com/chapter/10.1007/978-981-96-0570-5_14) | pure method | 4 | 4 | 4 | A | Explicitly enumerates two limitations and introduces modules in execution order. |
| 6 | MHHCR: Multi-behavior Heterogeneous Hypergraph Contrastive Recommendation | 2024 | WISE 2024, LNCS 15438 | [Springer](https://link.springer.com/chapter/10.1007/978-981-96-0570-5_7) | pure method | 3 | 4 | 4 | B | Compact 12-page example with code availability and direct challenge-to-module mapping. |
| 7 | Self-attention Convolutional Neural Network for Sequential Recommendation | 2023 | WISE 2023, LNCS 14306 | [Springer](https://link.springer.com/chapter/10.1007/978-981-99-7254-8_44) | pure method | 3 | 3 | 3 | B | Short sequential-recommendation paper; useful mainly for compactness. |
| 8 | Informative Anchor-Enhanced Heterogeneous Global Graph Neural Networks for Personalized Session-Based Recommendation | 2023 | WISE 2023, LNCS 14306 | [Springer](https://link.springer.com/chapter/10.1007/978-981-99-7254-8_45) | pure method | 4 | 4 | 4 | B | Demonstrates concise motivation around anonymity and long-distance item relations. |

## Writing Clusters

### Challenge-to-Module Abstracts

The WISE 2024 papers commonly state one or two concrete shortcomings, introduce the method name, describe modules in execution order, and finish with datasets and an effectiveness statement. LC-SOFT CRSID should follow this pattern while avoiding unsupported result claims before experiments are finalized.

### Auditable Method Sections

The fully accessible MDAP paper organizes Methodology as problem definition, framework overview, two mechanism subsections, objective function, and training algorithm. Each subsection introduces the module's purpose before equations and explains the role of the result afterward. This is directly applicable to LC-SOFT CRSID.

### Evidence Organization

The common experiment sequence is datasets, baselines, protocol, implementation, overall performance, and ablation. Results are discussed as explicit observations rather than a restatement of table values.

## Positioning for LC-SOFT CRSID

- Use WISE's application-oriented framing: explain why rigid SID assignments damage practical sequential recommendation, especially for sparse items.
- Keep RQ-KMeans concise because it is not the contribution.
- Give the Local-Consistent Soft SID construction the largest share of the Method section.
- Add an overview figure before detailed formulas.
- Tie each named module to an ablation and one reviewer-facing question.
- Treat code paths and implementation line numbers as internal notes, not paper content.

## Formatting Caution

Published WISE 2024 recommendation papers span roughly 12–16 proceedings pages, but this observation is not a submission rule. Use the official Springer LNCS template and verify the exact page limit and anonymization policy from the target-year WISE Call for Papers.
