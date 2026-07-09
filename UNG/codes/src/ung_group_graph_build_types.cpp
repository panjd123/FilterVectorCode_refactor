#include "include/ung_group_graph_build_types.h"

namespace ANNS
{

GroupGraphBuildContext make_group_graph_build_context(const UngBuildConfig &build_config,
                                                      IdxType max_degree,
                                                      uint32_t num_threads)
{
   GroupGraphBuildContext context;
   context.route = make_group_graph_route(build_config);
   context.cpu_settings = make_cpu_group_graph_settings(max_degree, num_threads);
   return context;
}

TagoreGroupBuildContext make_tagore_group_build_context(const UngBuildConfig &build_config,
                                                        IdxType max_degree,
                                                        uint32_t num_threads)
{
   TagoreGroupBuildContext context;
   context.route = make_group_graph_route(build_config);
   context.settings = make_tagore_group_graph_settings(max_degree,
                                                       num_threads,
                                                       context.route.is_adaptive_cuda());
   context.prune_mode = context.route.tagore_prune_mode();
   context.exact_batch_threshold = context.settings.exact_batch_threshold(context.prune_mode);
   context.runtime_config =
       make_tagore_cuda_runtime_config(build_config.tagore_k, static_cast<uint32_t>(max_degree), true);
   context.runtime_config.fast_exact_batch_threshold = context.exact_batch_threshold;
   return context;
}

} // namespace ANNS
