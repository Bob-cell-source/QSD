# Method-to-Code Map

| Paper component | Implementation |
|---|---|
| Metadata text construction | `lcsoftcrsid/semantic_id.py::build_item_text` |
| Residual K-Means Semantic ID | `lcsoftcrsid/semantic_id.py::residual_kmeans` |
| Slot-aware local neighborhood | `lcsoftcrsid/soft_sid.py::build_soft_sid_table` |
| Local support pruning and Top-M candidates | `lcsoftcrsid/soft_sid.py::build_soft_sid_table` |
| Soft SID reliability | `lcsoftcrsid/soft_sid.py::build_soft_sid_table` |
| Optional learnable candidate selection | `LCSoftCRSIDItemEncoder.candidate_weights` |
| Semantic basis | `LCSoftCRSIDItemEncoder.semantic_basis_embedding` |
| Shared semantic residual | `LCSoftCRSIDItemEncoder.shared_residual_embedding` |
| Private item residual | `LCSoftCRSIDItemEncoder.private_residual_embedding` |
| Reliability-aware residual calibration | `LCSoftCRSIDItemEncoder.forward` |
| Optional monotonic learnable calibration | `LCSoftCRSIDItemEncoder.residual_alpha` |
| Causal sequence modeling | `CausalTransformerEncoder` |
| Candidate matching | `LCSoftCRSID.forward` |
| Sampled recommendation loss | `trainer.py::sampled_cross_entropy` |
| Leave-one-out full-ranking evaluation | `trainer.py::evaluate_full_ranking` |

The paper method is represented by `LCSoftCRSID`. The item representation can
also be used independently through `LCSoftCRSIDItemEncoder`.
