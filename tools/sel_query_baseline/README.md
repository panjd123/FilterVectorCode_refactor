# Recall/time-advantage query selection

Build query tasks from an existing query task and per-query benchmark CSVs.

```bash
python3 tools/sel_query_baseline/select_query_by_recall_advantage.py \
  --config tools/sel_query_baseline/select_query_config.json
```

## Curator time-advantage mode

The default config now uses `selection_mode: curator_time_advantage` and builds:

- input task: `/home/dev/graphdb/FilterVectorData/Reviews/query_selected_recall_advantage`
- special-blocks results: `cpu_bruteforce_els_special_blocks/query_selected_recall_advantage_1000_1000_20000/results/query_details_repeat1.csv`
- Curator results: `Curator/query_selected_recall_advantage_search_ef64_search_ef10240/results/curator_results.csv`
- output task: `/home/dev/graphdb/FilterVectorData/Reviews/query_selected_recall_curator_time_advantage`

Selection rules:

- For each method and source query, keep the fastest row whose recall is at least `min_recall` (`0.95` by default). The special-blocks `Lsearch` and Curator `search_ef` do not need to be the same.
- Keep only queries where Curator time divided by special-blocks time is at least `min_speedup` (`2.0` by default), and the absolute time gap is at least `min_delta_ms`.
- Sort candidates by largest speedup, then fastest special-blocks time.
- Emit `target_num_queries` rows (`1000` by default). If there are fewer unique candidates, repeat the ranked candidates from fastest/best onward until the output reaches the target count.

The output `selected_queries.csv` includes the source query id, special `Lsearch`, Curator `curator_lsearch`, special and Curator timings, `speedup_vs_curator`, and the original rank before repetition.

## Original recall-advantage mode

Set `selection_mode` to `recall_advantage` to keep the previous behavior: select queries where `cpu_bruteforce_els_special_blocks` recall is no worse than `favor` at the first shared `Lsearch`, while the last special-blocks `Lsearch` recall stays above `min_last_special_recall`.

Set `overwrite` to `true` in the config to replace an existing output task.
