# Search Notes

## Search Mode

- Mode: standard novelty and prior-art scan.
- Date: 2026-06-09.
- Scope: recommender-system Semantic IDs, generative recommendation tokenizers, soft vector quantization, multiple identifiers, SID overlap/Hamming structure, and local-neighborhood code smoothing.

## Safe Queries Used

- semantic ID recommendation soft assignment residual quantization
- semantic IDs soft hard assignment recommender
- multi-slot overlap semantic ID recommendation
- Hamming neighborhood semantic ID recommendation
- semantic ID neighborhood overlap tokens soft distribution recommendation
- product quantization soft assignment multiple codewords local neighborhood
- multi-identifier item tokenization generative recommendation
- local consistent semantic ID recommendation
- slot-wise soft token distribution

## Sources Checked

- arXiv paper records and available full text
- NeurIPS proceedings
- PMLR proceedings
- OpenReview
- CVF Open Access
- ACM DOI/proceedings records
- Stable author/project paper pages where proceedings pages were unavailable

## Screening Summary

- More than 20 candidate records were screened.
- Thirteen papers were retained as direct threats, close methodological precedents, or required background.
- Off-topic uses of "multi-slot" in interface ranking, auctions, communications, and unrelated structured prediction were excluded.
- Generic hashing papers were excluded unless they clarified Hamming-neighborhood precedent; they did not implement the target SID mechanism.
- Policy-excluded sources encountered during broad search were omitted from the candidate and citation lists.
- Search snippets alone were not used to support the exact-method conclusion.

## Evidence Boundary

- Public-source evidence supports the statement that CapsID soft-routes to several semantic capsules and that QuaSID uses low-Hamming SID overlap for collision-aware repulsion.
- The statement that no exact prior method was found is a search result, not proof that no unpublished, non-English, patented, or unindexed method exists.
- CapsID, QuaSID, DRQ, DIGER, and ReSID are recent 2026 records; several remain preprints and should be described as such.

## Unknowns

- A dedicated patent search was not performed.
- Full-text inspection was limited where only abstract/proceedings metadata was available.
- No public paper was found under the exact names "Local-Consistent Soft SID" or "slot-wise soft token distribution" for Semantic-ID recommendation.
- The venue status of some 2026 preprints may change after this report date.

## Handoff Notes

- For writing: frame novelty around post-hoc, tokenizer-agnostic, discrete-neighborhood estimation rather than broad soft assignment.
- For idea optimization: strengthen the distinction from CapsID by emphasizing no tokenizer retraining, hard-token anchoring, sparse Top-M support, and reliability calibration.
- For experiments: include hard SID, global soft assignment/no local filter, overlap-threshold ablation, Top-M ablation, and a continuous-distance neighborhood alternative.
- For review: expect reviewers to ask why SID overlap is a valid neighborhood metric and whether local smoothing propagates collisions; provide neighborhood purity and reliability calibration analysis.

