# Fill Empty Base-Label Rows

`fill_empty_label_rows.py` repairs empty rows in a base-label file by copying
the immediately preceding non-empty label set. This is useful before building
the Trie special-block index, which does not accept empty terminal label sets.

## Check

```bash
python3 tools/fill_empty_label_rows.py \
  /home/dev/graphdb/FilterVectorData/Genome/Genome_base_labels.txt \
  --check
```

## Write A New File

```bash
python3 tools/fill_empty_label_rows.py \
  /home/dev/graphdb/FilterVectorData/Genome/Genome_base_labels.txt \
  --output /home/dev/graphdb/FilterVectorData/Genome/Genome_base_labels.filled.txt
```

## Modify In Place

```bash
python3 tools/fill_empty_label_rows/fill_empty_label_rows.py \
  /home/dev/graphdb/FilterVectorData/Genome/Genome_base_labels.txt \
  --in-place
```

After changing the labels, rebuild the ordinary UNG index so that
`group_id_to_label_set` and `group_id_to_range` match the repaired labels.
Then run the Trie block build with `build_ung: true` once, or set
`build_ung: false` only when the rebuilt UNG index already exists.
