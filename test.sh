cd /home/dev/graphdb/FilterVectorCode_refactor

# gpulock perf --wait-gpu-idle 0 -- \
#   ./run_cpu_special_blocks_experiment.sh \
#   /home/dev/graphdb/FilterVectorCode_refactor/experiments/cpu_special_blocks/Amazon_config_Trie_block.json

# python3 experiments/search_comparison/run_search_comparison.py \
#   experiments/search_comparison/config_amazon_special_block_cap16_best.json

# ./run_search_comparison_experiment.sh /home/dev/graphdb/FilterVectorCode_refactor/experiments/search_comparison/config_amazon_Trie_block_hybrid.json
python3 experiments/favor/run_favor_experiment.py experiments/favor/Amazon_config.json
# python3 experiments/acorn_baseline/run_acorn_baseline.py experiments/acorn_baseline/Amazon_config.json
experiments/curator_baseline/run_curator_baseline_with_env.sh experiments/curator_baseline/Amazon_config.json
# python3 experiments/navix_baseline/run_navix_baseline.py experiments/navix_baseline/Amazon_config.json