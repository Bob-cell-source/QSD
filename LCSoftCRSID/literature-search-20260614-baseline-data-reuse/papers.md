# Public Baseline Result Reuse Audit

## Scope

This note checks whether published results can be reused for the planned Amazon Office, Beauty, Sports, and Toys experiments. A result is considered directly reusable only when the dataset version, filtering, temporal split, candidate protocol, and metric definition match.

## Findings

| Method | Public evaluation datasets relevant to this paper | Protocol compatibility | Reuse decision |
|---|---|---|---|
| TIGER | Amazon Beauty, Sports and Outdoors, Toys and Games, May 1996 to July 2014 | Leave-one-out; Recall/NDCG at 5 and 10; item history truncated to 20 | Conditionally reusable after dataset statistics and candidate evaluation are matched |
| GRU4Rec / SASRec / BERT4Rec results reported by TIGER | Same three Amazon datasets | TIGER states these values were taken from the public S3-Rec results under consistent preprocessing | Conditionally reusable together with TIGER under the same protocol |
| UniSRec | Includes Amazon Office, but its Office split has 87,436 users and 25,986 items | Dataset statistics differ substantially from the local Office data: 4,905 users and 2,420 items | Not reusable |
| LLM-ESR | Yelp, Amazon Fashion, Amazon Beauty | The official implementation uses sampled evaluation with 100 negatives | Not reusable for the paper's full-catalog ranking table |
| LLMEmb | Yelp, Amazon Fashion, Amazon Beauty | The official implementation reports sampled evaluation with 100 negatives | Not reusable for full-catalog ranking |
| LLM2Rec | Amazon Reviews 2023 Games, Arts, Movies, Sports, Baby, and Goodreads | Full ranking, but the Sports data contains 13,952 items and 136,740 interactions and uses the 2023 Amazon release | Not reusable for the older Amazon Sports benchmark |
| LETTER | Public results are available, but no verified result was found under the exact four planned dataset/protocol combinations | Dataset/protocol match is unverified | Reproduce or omit from the numerical comparison |
| ETEGRec | Amazon Reviews 2023 Musical Instruments, Video Games, and Industrial and Scientific | Different categories and data release | Not reusable |
| SRA-CL | Yelp, Amazon Sports, Beauty, and Office | Leave-one-out and full-catalog ranking; Sports, Beauty, and Office statistics match the standard Amazon benchmark, and Office exactly matches the local data | Strongest available source of reported baseline results; reuse with an explicit reported-results note and configuration caveat |

## SRA-CL Audit

SRA-CL is substantially more compatible with the local evaluation than the original LLM-ESR and LLMEmb papers. It removes users and items with fewer than five interactions, uses the last interaction for testing and the second-to-last interaction for validation, and ranks against the whole item set without negative sampling. Its Office statistics exactly match the local dataset: 4,905 users, 2,420 items, and 53,258 interactions. It also uses the same Sports and Beauty statistics reported by TIGER.

SRA-CL reports five-run averages for GRU4Rec, SASRec, BERT4Rec, S3-Rec, CL4SRec, CoSeRec, ICLRec, DuoRec, MCLRec, ICSRec, LRD, RLMRec, and LLM-ESR. Therefore, it provides a useful common source for classical, contrastive-learning, and LLM-enhanced baselines on Sports, Beauty, and Office.

However, its model settings differ from the current local configuration: SRA-CL uses embedding dimension 64, maximum sequence length 20, and dropout 0.5, whereas the current LoCoRec runs use dimension 128, maximum length 50, and dropout 0.2. The discrepancy is empirically visible: SRA-CL reports SASRec Office NDCG@10 of 0.0348, while the local SASRec run obtains 0.0571. Thus, combining the local LoCoRec result with SRA-CL's baseline numbers would likely overstate the relative gain unless the configuration difference is disclosed or LoCoRec is rerun under the SRA-CL setting.

### SRA-CL Reported NDCG@10

| Method | Sports | Beauty | Office |
|---|---:|---:|---:|
| GRU4Rec | 0.0096 | 0.0137 | 0.0260 |
| SASRec | 0.0157 | 0.0336 | 0.0348 |
| BERT4Rec | 0.0189 | 0.0352 | 0.0376 |
| S3-Rec | 0.0204 | 0.0327 | 0.0426 |
| CL4SRec | 0.0189 | 0.0329 | 0.0322 |
| CoSeRec | 0.0244 | 0.0410 | 0.0412 |
| ICLRec | 0.0235 | 0.0396 | 0.0411 |
| DuoRec | 0.0242 | 0.0443 | 0.0519 |
| MCLRec | 0.0257 | 0.0442 | 0.0538 |
| ICSRec | 0.0243 | 0.0437 | 0.0540 |
| LRD | 0.0191 | 0.0294 | 0.0431 |
| RLMRec | 0.0238 | 0.0439 | 0.0496 |
| LLM-ESR | 0.0221 | 0.0435 | 0.0468 |
| SRA-CL | 0.0274 | 0.0469 | 0.0575 |

## TIGER Numbers Available Under Its Original Protocol

TIGER reports the following NDCG@10 values: Sports 0.0225, Beauty 0.0384, and Toys 0.0432. Its table also contains GRU4Rec, SASRec, and BERT4Rec results under the same three-dataset setup. These numbers should not be mixed with locally obtained results unless the local Beauty, Sports, and Toys statistics exactly match TIGER's 22,363/12,101, 35,598/18,357, and 19,412/11,924 user/item counts, respectively, and the evaluation semantics are confirmed.

## Recommended Use

Use published TIGER-table values only for the three matching Amazon 2014 categories and explicitly mark them as reported results. Use local runs for Office and for any method evaluated with a different dataset release or sampled-negative protocol. Avoid mixing sampled HR/NDCG values from LLM-ESR or LLMEmb with full-ranking metrics.

For SRA-CL, there are two defensible options. The stronger option is to rerun LoCoRec with dimension 64, maximum sequence length 20, and dropout 0.5, report five-seed averages, and compare against the SRA-CL table as reported results. The weaker but usable option is to retain the current LoCoRec configuration and cite the SRA-CL results in a separately labeled reported-baseline table, while avoiding direct relative-improvement claims.
