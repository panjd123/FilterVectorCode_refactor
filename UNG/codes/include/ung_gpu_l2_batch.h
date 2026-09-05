#ifndef ANNS_UNG_GPU_L2_BATCH_H
#define ANNS_UNG_GPU_L2_BATCH_H

#include "config.h"

#include <cstddef>

namespace ANNS
{

bool gpu_l2_batch_compute_float(const float *base_vectors,
                                IdxType num_points,
                                IdxType dim,
                                const float *query,
                                const IdxType *ids,
                                size_t count,
                                float *out_distances);

bool gpu_l2_batch_compute_query_candidates_float(const float *base_vectors,
                                                 IdxType num_points,
                                                 IdxType dim,
                                                 const float *query_vectors,
                                                 size_t query_count,
                                                 const IdxType *candidate_ids,
                                                 const IdxType *candidate_query_ids,
                                                 size_t candidate_count,
                                                 float *out_distances);

} // namespace ANNS

#endif // ANNS_UNG_GPU_L2_BATCH_H
