# Search Notes

## Purpose

Determine whether using item training frequency to calibrate shared versus item-specific representations is established prior art.

## Public Queries

- item frequency adaptive embedding recommendation long tail shared private representation
- popularity-aware gating item embedding recommender
- head tail item representation transfer frequency recommendation
- activity-aware gating semantic ID recommendation

## Evidence Boundary

- Directly verified from public paper pages: title, date/status, abstract-level mechanism, and venue where stated.
- Inference: papers using long-tail splits or popularity weighting support frequency as a supervision proxy, but do not necessarily use LC-SOFT CRSID's exact formula.
- No inspected earlier paper was found using the exact allocation \(f_i/(f_i+\tau\rho_i)\) with a local Semantic-ID reliability term.
- Absence from this targeted search is not proof of absolute novelty.
- AKT-Rec overlaps with the shared/individual representation and activity-dependent fusion principle, but not with LC-SOFT CRSID's local Soft-SID construction or complete recommendation pipeline.
- Local Git evidence: the broader low-frequency sharing investigation predates AKT-Rec's May 22, 2026 arXiv release; the currently identifiable explicit shared/private frequency-allocation implementation first appears in a June 1, 2026 commit. This is not evidence of public priority.

## Source Policy

Primary arXiv paper pages and venue metadata were used. Low-quality and policy-excluded sources were not included.
