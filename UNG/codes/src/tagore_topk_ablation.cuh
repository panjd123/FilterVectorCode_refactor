#pragma once

#include <cuda_fp16.h>

namespace ANNS
{
// Full-sort controls for the three GNN-Descent candidate-update kernels.
// These are separate kernels so the optimized kernels retain their original
// shared-memory footprint and launch resource requirements.
__global__ void tagore_sample_full_sort(unsigned *, unsigned *, unsigned,
                                      const half *, unsigned, unsigned, float *,
                                      float *, bool *, unsigned *, unsigned, unsigned);
__global__ void tagore_nn_descent_full_sort(unsigned *, unsigned *, unsigned, unsigned,
                                          float *, float *, bool *, unsigned *, unsigned);
__global__ void tagore_reverse_full_sort(unsigned *, unsigned *, float *, float *,
                                       unsigned *, unsigned);
}
