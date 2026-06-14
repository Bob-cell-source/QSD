# Search Notes

- Search purpose: determine whether recent sequential recommendation and Semantic-ID papers provide directly reusable benchmark results.
- Sources: original arXiv/ACM papers and official GitHub repositories.
- Matching rule: dataset name alone is insufficient; release, statistics, split, ranking candidates, and metrics must match.
- Local Office statistics checked from `runs/office/stats.json`: 4,905 users, 2,420 items, and 53,258 interactions.
- Main reusable source: TIGER, arXiv:2305.05065.
- Main incompatibilities: Amazon Reviews 2023 versus the older Amazon benchmark, and 100-negative sampled evaluation versus full-catalog ranking.
- SRA-CL is the closest public result source: exact Office statistics, matching leave-one-out split, and full-catalog ranking. Its architecture settings differ from the current LoCoRec runs, so values remain conditional rather than automatically interchangeable.
