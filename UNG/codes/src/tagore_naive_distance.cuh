#pragma once
#include <cuda_fp16.h>
namespace ANNS
{
// Scalar CUDA-Core local join for the same 16x16 candidate tiles as Tagore.
// Input vectors remain FP16; dot products accumulate in FP32, without WMMA.
__global__ void tagore_naive_local_join(const half *, const float *, unsigned *, float *,
                                      unsigned *, const unsigned *, const unsigned *,
                                      const unsigned *, unsigned);
}
