# Trained Hard SID gradient-conflict diagnostic

This is a read-only audit. No checkpoint is fine-tuned and no splitting method is added.

## Primary matched models

Use `consolidated_hard_scaled` and `consolidated_scaled` at seeds 2026/2027/2028 from `runs/office/consolidated_ablation_20260910`. These share the consolidated architecture, initialization rule, training protocol, and per-seed validation selection. They differ in Hard versus local Soft assignment. Neither the old paper numbers nor `full_scaled` versus a simplified model supply the Soft gain.

The gradient checkpoint is the validation-selected Hard model. Soft gains come from the paired existing test evaluations in their original user order. Input checkpoint and sequence hashes are stored. Gradients are collected only from training prefixes, never from validation/test targets. Associations with test gains are exploratory diagnostics, not a new hyperparameter-selection criterion.

## Exact meaning of item gradient

Define `L_i` as mean sampled CE across contexts whose **next-item target is i**. `g_i,c = d L_i / d E_c` includes all computational paths to the shared token: history, positive candidate, and negative candidates. Only rows corresponding to i's own Hard SID are retained for within-token comparisons. Each loss uses the original run's 100 eligible negatives and original score temperature.

This differs from grouping lookup-path gradients by the identity of the item looked up. That earlier refinement probe answers a different question and is not reused as if it computed target-conditioned `L_i`.

Two derivatives are reported:

1. `full`: the true target-conditioned CE derivative described above.
2. `positive`: the same CE's derivative through the positive item's lookup only, with upstream context and softmax coefficient detached. This is a partial derivative diagnostic, not an alternative full training loss. It controls for direct negative-item and history-path contributions, although CE competition still affects its scalar coefficient.

## Sampling and reproducibility

- Split complete user rows into two groups with fixed seed1707.
- Require at least4 training target examples per item in **each** user group.
- Sample up to16 contexts per item per group. Use item/view-specific seeds, independent of training seed, to match contexts and negative draws across checkpoints.
- The trained model has seen both user groups; this audits gradient reproducibility across different users, not unseen-user generalization.
- No dropout, optimizer update, auxiliary loss, or test target enters gradient computation.
- Aggregate each target's gradients before comparing distinct items. Exclude numerically zero vectors.
- Token statistics require at least4 eligible items; this restricts conclusions to supported parts of the catalog. Report both catalog occupancy and eligible coverage.

Pair conflict is cosine<0. Also report cosine<-.01 and the fraction negative by at least .01 in **both** independent user groups. Random cross-token pairs are an additional descriptive control, not frequency- or semantic-matched pairs.

## Semantic and behavioral evidence

- Semantic similarity: cosine of existing frozen item text embeddings, explicitly remapped through embedding_item_ids.json.
- Behavioral similarity: (a) cosine of binary training-user incidence vectors; (b) cosine of predecessor-item context profiles using the last5 training items, weighted by inverse lag.
- Both behavioral views use raw training interactions, not SID overlap or learned SID embeddings. They are not statistically independent of the recommender's training data.
- Among the top semantic-similarity quartile of within-token pair entries, compare zero observed behavioral overlap against the upper half of nonzero overlap. Zero overlap means no observed support; it is not proof of genuinely dissimilar preferences.
- Pair entries may repeat across levels/tokens and share items. Do not treat them as independent observations for significance tests.

## Soft gain and plots

For token c, group test users whose true next item uses c. Compute paired mean `NDCG_soft - NDCG_hard` in that group. Require20 test users for conflict/gain associations; report group sizes. Groups overlap across levels, so global correlations may have confounding and must not be read as causal transfer effects.

Report pooled and level-specific Spearman associations, and within-level partial rank correlations adjusting for occupancy, mean training frequency, test support, and gradient norm. These are descriptive correlations without naive pair-level significance claims.

Each seed produces PNG/PDF with occupancy/conflict scatter, conflict/Soft-gain scatter, and a cosine heatmap of the largest eligible token group (up to60 most frequent items). The heatmap group is selected by size, not by observed conflict. CSV contains all token statistics; NPZ contains gradients, pair entries and plotting matrices.

```bash
python -m LoCoRecSimple.gradient_conflict_diagnostic
```

Meaningful checks cover exact equality with the model's full CE gradient, averaging across target examples, positive-path locality, training-only behavioral extraction, and cosine signs.

Negative cosine alone does not establish harmful sharing: CE discrimination, sampling noise, optimization state, and different contexts can all cause disagreement. The requested mechanism story is supported only if reproducibility and independent behavior/performance associations actually align with it.
