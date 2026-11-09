#include "tagore_naive_distance.cuh"
#include <Tagore_src.cuh>

namespace ANNS
{
namespace
{
// Each thread scans one 16-candidate tile directly. The FP16 input, FP32
// norm-dot distance expression, invalid-distance thresholds, tile coverage and
// per-tile nearest-candidate policy match nn_descent_opt_cal. No tensor units,
// shared vector tiles, extra iterations or artificial delays are involved.
__device__ float direct_distance(const half *data, const float *power,
                                unsigned a, unsigned b, unsigned dim)
{
   float dot = 0.0f;
   for (unsigned d = 0; d < dim; ++d)
      dot = fmaf(__half2float(data[static_cast<size_t>(a) * dim + d]),
                 __half2float(data[static_cast<size_t>(b) * dim + d]), dot);
   if (dot < 0.01f)
      dot = -INF_DIS;
   const float distance = power[a] + power[b] - 2.0f * dot;
   return distance < 0.01f ? INF_DIS : distance;
}

__device__ void append_candidate(unsigned source, unsigned target, float distance,
                                 unsigned *reverse, float *distances, unsigned *counts)
{
   const unsigned position = atomicAdd(counts + source, 1u);
   if (position < RESERVENUM)
   {
      reverse[static_cast<size_t>(source) * RESERVENUM + position] = target;
      distances[static_cast<size_t>(source) * RESERVENUM + position] = distance;
   }
}
}

__global__ void tagore_naive_local_join(const half *data, const float *power,
                                      unsigned *reverse, float *distances, unsigned *counts,
                                      const unsigned *new_counts, const unsigned *old_counts,
                                      const unsigned *hybrid, unsigned dim)
{
   const unsigned anchor = blockIdx.x;
   const unsigned tid = threadIdx.x + threadIdx.y * blockDim.x;
   const unsigned threads = blockDim.x * blockDim.y;
   const unsigned nn = new_counts[anchor], no = old_counts[anchor];
   __shared__ unsigned new_ids[2 * SAMPLE], old_ids[2 * SAMPLE];
   for (unsigned i = tid; i < 2 * SAMPLE; i += threads)
   {
      new_ids[i] = i < nn ? hybrid[anchor * (4 * SAMPLE) + i] : 0;
      old_ids[i] = i < no ? hybrid[anchor * (4 * SAMPLE) + 2 * SAMPLE + i] : 0;
   }
   __syncthreads();
   constexpr unsigned tiles = 2 * SAMPLE / BLK_H;
   constexpr unsigned jobs = 2 * SAMPLE * tiles;
   // New-to-old and new-to-new row minima. Tagore's new-new stage visits only
   // the upper triangle of 16x16 tiles, including the diagonal.
   for (unsigned job = tid; job < jobs; job += threads)
   {
      const unsigned row = job / tiles, tile = job % tiles;
      if (row >= nn)
         continue;
      const unsigned source = new_ids[row];
      if (no > 0)
      {
         float best = 3.402823466e+38F;
         unsigned target = 0;
         for (unsigned j = 0; j < BLK_H; ++j)
         {
            const unsigned col = tile * BLK_H + j;
            const float distance = col < no ? direct_distance(data, power, source, old_ids[col], dim) : power[source] + power[0] + 2.0f * INF_DIS;
            if (distance <= best) { best = distance; target = old_ids[col]; }
         }
         if (best < INF_DIS)
            append_candidate(source, target, best, reverse, distances, counts);
      }
      if (row / BLK_H <= tile)
      {
         float best = 3.402823466e+38F;
         unsigned target = 0;
         for (unsigned j = 0; j < BLK_H; ++j)
         {
            const unsigned col = tile * BLK_H + j;
            const float distance = col < nn ? direct_distance(data, power, source, new_ids[col], dim) : power[source] + power[0] + 2.0f * INF_DIS;
            if (distance <= best) { best = distance; target = new_ids[col]; }
         }
         // Empty tiles retain Tagore's norm + 2*INF sentinel, which sorts
         // after existing INF entries. Do not collapse it to INF_DIS.
         append_candidate(source, target, best, reverse, distances, counts);
      }
   }
   // Old-to-new column minima from the same new-old tiles. Keeping this as a
   // direct scan deliberately avoids the matrix-result reuse of the WMMA path.
   for (unsigned job = tid; job < jobs; job += threads)
   {
      const unsigned row = job / tiles, tile = job % tiles;
      if (row >= no)
         continue;
      const unsigned source = old_ids[row];
      float best = 3.402823466e+38F;
      unsigned target = 0;
      for (unsigned j = 0; j < BLK_H; ++j)
      {
         const unsigned col = tile * BLK_H + j;
         const float distance = col < nn ? direct_distance(data, power, source, new_ids[col], dim) : power[source] + power[0] + 2.0f * INF_DIS;
         if (distance <= best) { best = distance; target = new_ids[col]; }
      }
      if (best < INF_DIS)
         append_candidate(source, target, best, reverse, distances, counts);
   }
}
}
