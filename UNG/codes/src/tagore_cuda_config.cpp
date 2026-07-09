#include "include/tagore_graph_builder.h"

#include <algorithm>
#include <cstdlib>

namespace ANNS
{
namespace
{

uint32_t env_uint(const char *name, uint32_t default_value, uint32_t min_value, uint32_t max_value)
{
   const char *value = std::getenv(name);
   if (!value || !*value)
      return default_value;
   char *end = nullptr;
   const unsigned long parsed = std::strtoul(value, &end, 10);
   if (end == value)
      return default_value;
   return static_cast<uint32_t>(std::min<unsigned long>(std::max<unsigned long>(parsed, min_value), max_value));
}

TagoreFastExactBatchConfig make_fast_exact_batch_config(uint32_t k, uint32_t final_degree)
{
   TagoreFastExactBatchConfig config;
   const uint32_t default_head_keep = final_degree > 8 ? final_degree - 8 : final_degree;
   config.head_keep = env_uint("UNG_FAST_GRNND_LIGHT_HEAD", default_head_keep, 1, final_degree);
   config.anchor_tail = env_uint("UNG_FAST_EXACT_ANCHOR_TAIL", 1, 0, 1);
   config.anchor_slots = env_uint("UNG_FAST_EXACT_ANCHOR_SLOTS", 0, 0, final_degree);
   config.bidir_anchor = env_uint("UNG_FAST_EXACT_BIDIR_ANCHOR", 0, 0, 1);
   config.reverse_cap = env_uint("UNG_FAST_EXACT_REVERSE_CAP", 0, 0, final_degree);
   config.reverse_slots = env_uint("UNG_FAST_EXACT_REVERSE_SLOTS", 0, 0, final_degree);
   const uint32_t default_reverse_forward_cap = std::min<uint32_t>(k > 0 ? k - 1 : 1, final_degree);
   config.reverse_forward_cap = env_uint("UNG_FAST_EXACT_REVERSE_FORWARD_CAP",
                                         default_reverse_forward_cap, 1, k > 0 ? k - 1 : 1);
   config.pinned_host = env_uint("UNG_FAST_EXACT_PINNED_HOST", 0, 0, 1) != 0;
   config.device_lookup = env_uint("UNG_FAST_EXACT_DEVICE_LOOKUP", 1, 0, 1) != 0;
   config.direct_h2d_requested = env_uint("UNG_FAST_EXACT_DIRECT_H2D", 1, 0, 1) != 0;
   config.direct_h2d_max_runs = env_uint("UNG_FAST_EXACT_DIRECT_H2D_MAX_RUNS", 4096, 1, 1u << 20);
   config.graph_stride = env_uint("UNG_FAST_EXACT_COMPACT_D2H", 1, 0, 1) != 0 ? final_degree + 1 : k;
   config.use_warp_kernel = env_uint("UNG_FAST_EXACT_WARP_KERNEL", 0, 0, 1) != 0;
   return config;
}

TagoreFastGrnndPruneConfig make_fast_grnnd_prune_config(uint32_t k, uint32_t final_degree)
{
   TagoreFastGrnndPruneConfig config;
   config.light_threshold = env_uint("UNG_FAST_GRNND_LIGHT_PRUNE_NX", 256, 0, 1u << 20);
   const uint32_t default_head_keep = final_degree > 8 ? final_degree - 8 : final_degree;
   config.light_head_keep = env_uint("UNG_FAST_GRNND_LIGHT_HEAD", default_head_keep, 1, final_degree);
   config.light_reverse_cap = env_uint("UNG_FAST_GRNND_LIGHT_REVERSE_CAP", 0, 0, final_degree);
   config.repair_degree = env_uint("UNG_FAST_GRNND_REPAIR_DEGREE", 1, 0, 1);
   const uint32_t default_reverse_slots =
       std::min<uint32_t>(config.light_reverse_cap, final_degree > config.light_head_keep ? final_degree - config.light_head_keep : 1);
   config.light_reverse_slots = env_uint("UNG_FAST_GRNND_LIGHT_REVERSE_SLOTS",
                                         default_reverse_slots, 1, final_degree);
   const uint32_t default_light_forward_cap = std::min<uint32_t>(k > 0 ? k - 1 : 1,
                                                                 std::max<uint32_t>(1, config.light_head_keep));
   config.light_reverse_forward_cap = env_uint("UNG_FAST_GRNND_LIGHT_REVERSE_FORWARD_CAP",
                                               default_light_forward_cap, 1, k > 0 ? k - 1 : 1);
   const uint32_t default_forward_cap = std::min<uint32_t>(k > 0 ? k - 1 : 1,
                                                           std::max<uint32_t>(final_degree, 2u * final_degree));
   config.forward_cap = std::min<uint32_t>(
       k > 0 ? k - 1 : 1,
       env_uint("UNG_FAST_GRNND_FORWARD_CAP", default_forward_cap, 1, k > 0 ? k - 1 : 1));
   config.reverse_cap = env_uint("UNG_FAST_GRNND_REVERSE_CAP",
                                 std::max<uint32_t>(1, final_degree), 0, final_degree);
   return config;
}

} // namespace

TagoreCudaRuntimeConfig make_tagore_cuda_runtime_config(uint32_t k,
                                                        uint32_t final_degree,
                                                        bool allow_parallel)
{
   if (k <= final_degree)
      k = final_degree + 1;

   TagoreCudaRuntimeConfig config;
   config.fast_exact = make_fast_exact_batch_config(k, final_degree);
   config.fast_prune = make_fast_grnnd_prune_config(k, final_degree);
   config.fast_exact_batch_threshold = env_uint("UNG_FAST_GRNND_BATCH_EXACT_NX", 0, 0, 1u << 20);
   config.requested_streams = allow_parallel ? env_uint("UNG_TAGORE_BATCH_STREAMS", 1, 1, 16) : 1;
   config.compact_d2h_requested = env_uint("UNG_TAGORE_COMPACT_D2H", 1, 0, 1) != 0;
   return config;
}

} // namespace ANNS
