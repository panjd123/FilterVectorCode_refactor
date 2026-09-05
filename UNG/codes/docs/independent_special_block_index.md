# Independent Trie Block Index

UNG and the Trie block index have separate build and persistence lifecycles.

## 1. Build UNG

Run `build_UNG_index` normally. `UNG_SPECIAL_BLOCKS` is ignored by this
executable, and the UNG output contains no `special_block_*` files or metadata.

## 2. Build the Trie block index

```bash
build_special_block_index \
  --ung_index_path_prefix /path/to/ung/index_files/ \
  --block_index_path_prefix /path/to/block/index_files/ \
  --result_path_prefix /path/to/block/results/ \
  --data_type float \
  --dist_fn L2 \
  --num_threads 60 \
  --min_points 1000 \
  --max_degree 32 \
  --num_cross_edges 4 \
  --Lbuild 100 \
  --alpha 1.2
```

The builder reads only these files from the UNG index:

- `meta`
- `vecs.bin`
- `labels.txt`
- `group_id_to_label_set`
- `group_id_to_range`

It writes a self-contained block sidecar directory containing block metadata,
the block Trie, regular Trie edges, special edges, and its own `meta` file.
Binary special-edge output is the default. Set `UNG_SPECIAL_EDGE_BINARY_ONLY=0`
when edge CSV diagnostics are required, and set
`UNG_SPECIAL_BLOCK_METADATA_CSV=1` when the three block metadata CSV files are
required for inspection.

Edge files use CSR v2. Source IDs are represented once by the offset table;
regular edges store only a 32-bit target, and special edges store a 32-bit
target plus a packed block/kind word. The loader remains compatible with the
legacy 16-byte-per-edge format.

The independent `meta` and `build_time.csv` report total build time, exact
on-disk index bytes, build-time expanded memory, and final loaded CSR memory.
Logical memory uses `sizeof(T) * size()`; allocated memory includes container
objects plus `sizeof(T) * capacity()`. The builder releases its expanded edge
rows and reloads the saved CSR once so the `loaded_memory_*` metrics describe
the actual search representation. Loading also prints its measured time and
the two recomputed memory sizes.
CSR v2 indexes remain flat after loading: search reads each source range
directly from offsets and contiguous ID/payload arrays instead of expanding
the file into one `std::vector` per point. Legacy v1 and CSV indexes retain the
older expanded fallback path.

## 3. Search with the optional block index

Pass both index paths to `search_UNG_index`:

```bash
search_UNG_index \
  --index_path_prefix /path/to/ung/index_files/ \
  --block_index_path_prefix /path/to/block/index_files/ \
  --entry_group_provider special_block_trie \
  ...
```

Omit `--block_index_path_prefix` for plain UNG search. Existing legacy indexes
that store block files in the UNG directory remain readable.
