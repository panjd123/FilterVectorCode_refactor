# Generate Base Labels

This experiment runs the UNG C++ tool `tools/generate_base_labels` from a JSON
configuration file.

Run from the repository root:

```bash
python3 experiments/generate_base_labels/run_generate_base_labels.py
```

Use a different config:

```bash
python3 experiments/generate_base_labels/run_generate_base_labels.py path/to/config.json
```

Preview the command without generating labels:

```bash
python3 experiments/generate_base_labels/run_generate_base_labels.py --dry-run
```

The default config writes one base-label file for each configured dataset:

```text
/home/dev/graphdb/FilterVectorData/Amazon/Amazon_base_labels.txt
/home/dev/graphdb/FilterVectorData/Reviews/Reviews_base_labels.txt
/home/dev/graphdb/FilterVectorData/VariousImg/VariousImg_base_labels.txt
/home/dev/graphdb/FilterVectorData/Genome/Genome_base_labels.txt
```

Config fields:

- `build_dir`: CMake build directory containing `tools/generate_base_labels`.
- `datasets`: Optional list of dataset jobs. Each job can set `dataset`, `output_file`, and `num_points`, and inherits the top-level defaults below.
- `output_file`: Label text file to create. Required either at the top level for a single job, or inside each `datasets` entry.
- `num_points`: Number of base vectors, one output line per vector. Required either at the top level for a single job, or inside each `datasets` entry.
- `num_labels`: Number of unique labels.
- `distribution_type`: One of `zipf`, `multi_normial`, `uniform`, `poisson`, or `one_per_point`.
- `expected_num_label`: Expected labels per vector for the relevant distributions.
- `max_num_label`: Passed through to the C++ tool for compatibility.

`zipf` is the recommended default for filtered-vector experiments because it
creates a long-tailed label frequency distribution.
