#ifndef ANNS_UNG_CROSS_EDGE_RESULT_H
#define ANNS_UNG_CROSS_EDGE_RESULT_H

#include "ung_cross_edge_config.h"

#include <cstddef>
#include <string>
#include <vector>

namespace ANNS
{

class SearchQueue;

struct CrossEdgeBuildTiming
{
   double generate_ms = 0.0;
   double additional_ms = 0.0;
   double merge_cross_ms = 0.0;
   double merge_additional_ms = 0.0;
   double add_offset_ms = 0.0;
   double output_storage_init_ms = 0.0;
   double additional_storage_init_ms = 0.0;
   double search_cache_init_ms = 0.0;
   double gpu_h2d_ms = 0.0;
   double gpu_kernel_ms = 0.0;
   double gpu_d2h_ms = 0.0;
};

struct CrossEdgeBuildResult
{
   CrossEdgeBuildTiming timing;
   CrossEdgeGpuWritebackMode active_writeback_mode = CrossEdgeGpuWritebackMode::SearchQueue;
   std::string route_summary;
   std::size_t flat_id_items = 0;
   bool gpu_backend_used = false;
   bool gpu_fallback_used = false;
   bool additional_direct_append = false;

   const char *active_writeback_mode_name() const
   {
      return cross_edge_gpu_writeback_mode_name(active_writeback_mode);
   }
};

struct CrossEdgeHostOutputView
{
   CrossEdgeGpuWritebackMode mode = CrossEdgeGpuWritebackMode::SearchQueue;
   const std::vector<SearchQueue> *search_queues = nullptr;
   const std::vector<std::vector<IdxType>> *id_vectors = nullptr;
   const std::vector<IdxType> *flat_ids = nullptr;
   IdxType topk = 0;

   bool uses_search_queues() const
   {
      return mode == CrossEdgeGpuWritebackMode::SearchQueue;
   }

   bool uses_id_vectors() const
   {
      return mode == CrossEdgeGpuWritebackMode::IdVector;
   }

   bool uses_flat_ids() const
   {
      return mode == CrossEdgeGpuWritebackMode::FlatId;
   }
};

} // namespace ANNS

#endif // ANNS_UNG_CROSS_EDGE_RESULT_H
