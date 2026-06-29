# Search Notes

## Safe Queries Used

- WISE 2025 sequential recommendation
- WISE 2024 recommendation semantic contrastive
- WISE 2023 sequential recommendation
- WISE recommendation long-tail and data sparsity

## Sources Checked

- Official Springer WISE conference series page.
- Official WISE 2021, 2023, 2024, and 2025 LNCS proceedings pages.
- Official Springer chapter pages and DOI metadata for shortlisted papers.

## Excluded Sources

- Policy-excluded and low-confidence sources were not used in the final candidate set.
- Papers were excluded when only a search snippet was available or when their task was too far from sequential recommendation and item representation.

## Unknowns

- Full experimental tables were not accessible for all subscription chapters; numeric-evidence scores are therefore conservative and based on inspectable abstracts.
- Citation counts were not used as a quality criterion.

## Handoff Notes

- For writing: group NCL4Rec, CSA4Rec, and SDARec by the level at which they introduce augmentation, then contrast them with LoCoRec's item-representation-level SID sharing calibration.
- For experiments: none of the three papers is automatically a required baseline because their training objectives and evaluation protocols differ from LoCoRec.
- For review: avoid describing these papers as hard-SID collision or quantization methods.
