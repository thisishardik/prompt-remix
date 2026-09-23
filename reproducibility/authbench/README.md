# AuthBench paper cohorts

The two JSON files in this directory contain the exact AuthBench `documentID`
values used in the PromptRemix paper:

- `canonical_query_ids.json`: 1,317 queries
- `short_text_query_ids.json`: 2,321 queries

Each file records language-level totals and the ordered IDs for every
language--genre cell.

Given a local copy of AuthBench, verify that all frozen IDs are available:

```bash
python scripts/5-evaluation/reproduce_authbench_samples.py \
  --authbench-dir /path/to/authbench \
  --verify-only
```

To extract the exact query records and recreate the per-cell sampled-ID files:

```bash
python scripts/5-evaluation/reproduce_authbench_samples.py \
  --authbench-dir /path/to/authbench \
  --output-dir authbench_promptremix_samples
```

The AuthBench directory must contain files at
`<language>/per_genre/authbench_<language>_<genre>_queries.jsonl`.
