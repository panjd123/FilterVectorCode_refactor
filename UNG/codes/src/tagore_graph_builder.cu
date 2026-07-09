#include "include/tagore_graph_builder.h"

#include <Tagore_src.cuh>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace ANNS
{
namespace
{

__device__ funcFormat tagore_dis_filter = distance_filter;

void ck(cudaError_t err, const char *what)
{
   if (err != cudaSuccess)
      throw std::runtime_error(std::string("TagoreCuda CUDA error at ") + what + ": " + cudaGetErrorString(err));
}

float compute_norm_factor(const float *data, uint32_t num_points, uint32_t dim)
{
   if (num_points == 0 || dim == 0)
      return 1.0f;
   float tmp_ave = 0.0f;
   for (uint32_t i = 0; i < dim; ++i)
      tmp_ave += std::abs(data[i]);
   tmp_ave /= static_cast<float>(dim);
   float norm_factor = 1.0f;
   while (tmp_ave < 0.5f && tmp_ave > 0.0f)
   {
      tmp_ave *= 10.0f;
      norm_factor *= 10.0f;
   }
   while (tmp_ave > 5.0f)
   {
      tmp_ave /= 10.0f;
      norm_factor /= 10.0f;
   }
   return norm_factor;
}

double elapsed_ms(std::chrono::high_resolution_clock::time_point start)
{
   return std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start).count();
}

float cuda_event_elapsed_ms(cudaEvent_t start, cudaEvent_t stop, const char *what)
{
   float ms = 0.0f;
   ck(cudaEventElapsedTime(&ms, start, stop), what);
   return ms;
}

TagoreCudaRuntimeConfig make_single_stream_config(TagoreCudaRuntimeConfig config)
{
   config.requested_streams = 1;
   return config;
}

__global__ void convert_float_to_half_kernel(const float *data, half *data_half, size_t total, float norm_factor)
{
   const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
   const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
   for (size_t i = tid; i < total; i += stride)
      data_half[i] = __float2half(data[i] * norm_factor);
}

__global__ void compact_graph_rows_kernel(const unsigned *src_graph,
                                          uint32_t *dst_graph,
                                          uint32_t num_points,
                                          uint32_t src_stride,
                                          uint32_t dst_stride,
                                          uint32_t final_degree)
{
   const uint32_t row_id = blockIdx.x;
   const uint32_t tid = threadIdx.x;
   if (row_id >= num_points)
      return;

   const unsigned *src = src_graph + static_cast<size_t>(row_id) * src_stride;
   uint32_t *dst = dst_graph + static_cast<size_t>(row_id) * dst_stride;
   const uint32_t degree = min(static_cast<uint32_t>(src[0]), final_degree);
   if (tid == 0)
      dst[0] = degree;
   for (uint32_t j = tid; j < final_degree; j += blockDim.x)
      dst[1 + j] = j < degree ? src[1 + j] : row_id;
}

__global__ void compute_center_half_kernel(const float *data, half *center_half,
                                           uint32_t num_points, uint32_t dim, float norm_factor)
{
   const uint32_t d = blockIdx.x;
   float sum = 0.0f;
   for (uint32_t i = threadIdx.x; i < num_points; i += blockDim.x)
      sum += data[static_cast<size_t>(i) * dim + d];

   __shared__ float partial[256];
   partial[threadIdx.x] = sum;
   __syncthreads();

   for (uint32_t offset = blockDim.x / 2; offset > 0; offset >>= 1)
   {
      if (threadIdx.x < offset)
         partial[threadIdx.x] += partial[threadIdx.x + offset];
      __syncthreads();
   }

   if (threadIdx.x == 0)
      center_half[d] = __float2half((partial[0] / static_cast<float>(num_points)) * norm_factor);
}

__device__ __forceinline__ float l2_half_distance_device(const half *values,
                                                         unsigned a,
                                                         unsigned b,
                                                         unsigned dim)
{
   float acc = 0.0f;
   for (unsigned d = threadIdx.x; d < dim; d += blockDim.x)
   {
      const float va = __half2float(values[static_cast<size_t>(a) * dim + d]);
      const float vb = __half2float(values[static_cast<size_t>(b) * dim + d]);
      const float diff = va - vb;
      acc += diff * diff;
   }
   for (int offset = 16; offset > 0; offset >>= 1)
      acc += __shfl_down_sync(0xffffffff, acc, offset);
   return acc;
}

__global__ void build_sampled_reverse_kernel(const unsigned *graph,
                                             unsigned *sampled_reverse,
                                             unsigned *sampled_reverse_num,
                                             unsigned num_points,
                                             unsigned k,
                                             unsigned outgoing_sample_limit,
                                             unsigned reverse_cap)
{
   const unsigned src = blockIdx.x * blockDim.x + threadIdx.x;
   if (src >= num_points)
      return;
   const unsigned degree = min(graph[static_cast<size_t>(src) * k], k > 0 ? k - 1 : 0);
   const unsigned limit = min(degree, outgoing_sample_limit);
   for (unsigned j = 0; j < limit; ++j)
   {
      const unsigned dst = graph[static_cast<size_t>(src) * k + 1 + j];
      if (dst >= num_points || dst == src)
         continue;
      const unsigned pos = atomicAdd(&sampled_reverse_num[dst], 1);
      if (pos < reverse_cap)
         sampled_reverse[static_cast<size_t>(dst) * reverse_cap + pos] = src;
   }
}

__global__ void grnnd_like_refine_kernel(unsigned *graph,
                                         const half *values,
                                         const unsigned *sampled_reverse,
                                         const unsigned *sampled_reverse_num,
                                         unsigned num_points,
                                         unsigned dim,
                                         unsigned k,
                                         unsigned final_degree,
                                         unsigned reverse_cap,
                                         float alpha)
{
   const unsigned src = blockIdx.x;
   const unsigned tid = threadIdx.x;
   if (src >= num_points)
      return;

   __shared__ unsigned cand[256];
   __shared__ float cand_dist[256];
   __shared__ unsigned cand_count;
   __shared__ unsigned final_nei[FINAL_DEGREE_SIZE];
   __shared__ unsigned final_count;
   __shared__ unsigned scan_pos;

   if (tid == 0)
      cand_count = 0;
   __syncthreads();

   const unsigned degree = min(graph[static_cast<size_t>(src) * k], k > 0 ? k - 1 : 0);
   for (unsigned j = tid; j < degree; j += blockDim.x)
   {
      const unsigned dst = graph[static_cast<size_t>(src) * k + 1 + j];
      if (dst < num_points && dst != src)
      {
         const unsigned pos = atomicAdd(&cand_count, 1);
         if (pos < 256)
            cand[pos] = dst;
      }
   }

   const unsigned rev_count = min(sampled_reverse_num[src], reverse_cap);
   for (unsigned j = tid; j < rev_count; j += blockDim.x)
   {
      const unsigned dst = sampled_reverse[static_cast<size_t>(src) * reverse_cap + j];
      if (dst < num_points && dst != src)
      {
         const unsigned pos = atomicAdd(&cand_count, 1);
         if (pos < 256)
            cand[pos] = dst;
      }
   }
   __syncthreads();

   if (tid == 0 && cand_count > 256)
      cand_count = 256;
   __syncthreads();

   for (unsigned i = 0; i < cand_count; ++i)
   {
      const float dist = l2_half_distance_device(values, src, cand[i], dim);
      if (tid == 0)
         cand_dist[i] = dist;
      __syncthreads();
   }

   bitonic_sort_id_and_dis(cand_dist, cand, cand_count);

   if (tid == 0)
   {
      final_count = 0;
      scan_pos = 0;
   }
   __syncthreads();

   while (final_count < final_degree && scan_pos < cand_count)
   {
      if (tid == 0)
      {
         while (scan_pos < cand_count && (cand[scan_pos] == src || cand[scan_pos] == 0xffffffffu))
            ++scan_pos;
         if (scan_pos < cand_count)
         {
            final_nei[final_count] = cand[scan_pos];
            cand[scan_pos] = 0xffffffffu;
            ++final_count;
            ++scan_pos;
         }
      }
      __syncthreads();
      if (final_count == 0 || scan_pos >= cand_count)
         break;

      const unsigned pivot = final_nei[final_count - 1];
      for (unsigned i = scan_pos; i < cand_count; ++i)
      {
         if (cand[i] == 0xffffffffu || cand[i] == src || cand[i] == pivot)
         {
            if (tid == 0)
               cand[i] = 0xffffffffu;
            __syncthreads();
            continue;
         }
         const float dij = l2_half_distance_device(values, pivot, cand[i], dim);
         if (tid == 0 && alpha * dij < cand_dist[i])
            cand[i] = 0xffffffffu;
         __syncthreads();
      }
   }

   for (unsigned i = tid; i < final_degree; i += blockDim.x)
      graph[static_cast<size_t>(src) * k + 1 + i] = i < final_count ? final_nei[i] : src;
   if (tid == 0)
      graph[static_cast<size_t>(src) * k] = final_count;
}

__global__ void fast_grnnd_prune_kernel(unsigned *graph,
                                        const half *values,
                                        const float *nei_distance,
                                        const unsigned *sampled_reverse,
                                        const unsigned *sampled_reverse_num,
                                        unsigned num_points,
                                        unsigned dim,
                                        unsigned k,
                                        unsigned final_degree,
                                        unsigned forward_cap,
                                        unsigned reverse_cap,
                                        float alpha)
{
   const unsigned src = blockIdx.x;
   const unsigned tid = threadIdx.x;
   if (src >= num_points)
      return;

   __shared__ unsigned cand[128];
   __shared__ float cand_dist[128];
   __shared__ unsigned cand_count;
   __shared__ unsigned final_nei[FINAL_DEGREE_SIZE];
   __shared__ unsigned final_count;

   if (tid == 0)
      cand_count = 0;
   __syncthreads();

   const unsigned out_degree = min(graph[static_cast<size_t>(src) * k], k > 0 ? k - 1 : 0);
   const unsigned out_limit = min(out_degree, forward_cap);
   for (unsigned j = tid; j < out_limit; j += blockDim.x)
   {
      const unsigned dst = graph[static_cast<size_t>(src) * k + 1 + j];
      if (dst < num_points && dst != src)
      {
         const unsigned pos = atomicAdd(&cand_count, 1);
         if (pos < 128)
            cand[pos] = dst;
      }
   }

   const unsigned rev_count = min(sampled_reverse_num[src], reverse_cap);
   for (unsigned j = tid; j < rev_count; j += blockDim.x)
   {
      const unsigned dst = sampled_reverse[static_cast<size_t>(src) * reverse_cap + j];
      if (dst < num_points && dst != src)
      {
         const unsigned pos = atomicAdd(&cand_count, 1);
         if (pos < 128)
            cand[pos] = dst;
      }
   }
   __syncthreads();

   if (tid == 0 && cand_count > 128)
      cand_count = 128;
   __syncthreads();

   for (unsigned i = 0; i < cand_count; ++i)
   {
      const unsigned dst = cand[i];
      const float dist = dst < num_points ? l2_half_distance_device(values, src, dst, dim) : 3.402823466e+38F;
      if (tid == 0)
         cand_dist[i] = dist;
      __syncthreads();
   }

   bitonic_sort_id_and_dis(cand_dist, cand, cand_count);
   for (unsigned i = tid + 1; i < cand_count; i += blockDim.x)
      if (cand[i] == cand[i - 1])
         cand[i] = 0xffffffffu;
   __syncthreads();

   if (tid == 0)
      final_count = 0;
   __syncthreads();

   for (unsigned scan = 0; scan < cand_count && final_count < final_degree; ++scan)
   {
      const unsigned dst = cand[scan];
      if (dst >= num_points || dst == src || dst == 0xffffffffu)
         continue;

      __shared__ unsigned reject;
      if (tid == 0)
         reject = 0;
      __syncthreads();

      for (unsigned j = 0; j < final_count; ++j)
      {
         const unsigned pivot = final_nei[j];
         const float dij = l2_half_distance_device(values, pivot, dst, dim);
         if (tid == 0 && alpha * dij < cand_dist[scan])
            reject = 1;
         __syncthreads();
         if (reject)
            break;
      }

      if (tid == 0 && !reject)
         final_nei[final_count++] = dst;
      __syncthreads();
   }

   for (unsigned i = tid; i < final_degree; i += blockDim.x)
      graph[static_cast<size_t>(src) * k + 1 + i] = i < final_count ? final_nei[i] : src;
   if (tid == 0)
      graph[static_cast<size_t>(src) * k] = final_count;
}

__global__ void fast_grnnd_light_prune_kernel(unsigned *graph,
                                              unsigned num_points,
                                              unsigned k,
                                              unsigned final_degree,
                                              unsigned head_keep,
                                              unsigned repair_degree)
{
   const unsigned src = blockIdx.x;
   const unsigned tid = threadIdx.x;
   if (src >= num_points)
      return;

   __shared__ unsigned final_nei[FINAL_DEGREE_SIZE];
   __shared__ unsigned final_count;
   if (tid == 0)
   {
      final_count = 0;
      const unsigned degree = min(graph[static_cast<size_t>(src) * k], k > 0 ? k - 1 : 0);
      const unsigned near_limit = min(degree, min(head_keep, final_degree));
      for (unsigned scan = 0; scan < near_limit && final_count < final_degree; ++scan)
      {
         const unsigned dst = graph[static_cast<size_t>(src) * k + 1 + scan];
         if (dst >= num_points || dst == src)
            continue;
         bool dup = false;
         for (unsigned j = 0; j < final_count; ++j)
         {
            if (final_nei[j] == dst)
            {
               dup = true;
               break;
            }
         }
         if (!dup)
            final_nei[final_count++] = dst;
      }
      if (final_count < final_degree)
      {
         const unsigned remaining = final_degree - final_count;
         const unsigned tail_start = near_limit;
         const unsigned tail_count = degree > tail_start ? degree - tail_start : 0;
         const unsigned step = tail_count > remaining ? (tail_count + remaining - 1) / remaining : 1;
         for (unsigned scan = tail_start; scan < degree && final_count < final_degree; scan += step)
         {
            const unsigned dst = graph[static_cast<size_t>(src) * k + 1 + scan];
            if (dst >= num_points || dst == src)
               continue;
            bool dup = false;
            for (unsigned j = 0; j < final_count; ++j)
            {
               if (final_nei[j] == dst)
               {
                  dup = true;
                  break;
               }
            }
            if (!dup)
               final_nei[final_count++] = dst;
         }
      }
      for (unsigned scan = near_limit; scan < degree && final_count < final_degree; ++scan)
      {
         const unsigned dst = graph[static_cast<size_t>(src) * k + 1 + scan];
         if (dst >= num_points || dst == src)
            continue;
         bool dup = false;
         for (unsigned j = 0; j < final_count; ++j)
         {
            if (final_nei[j] == dst)
            {
               dup = true;
               break;
            }
         }
         if (!dup)
            final_nei[final_count++] = dst;
      }
      if (repair_degree && num_points > 1)
      {
         for (unsigned hop = 1; hop < num_points && final_count < final_degree; ++hop)
         {
            const unsigned dst = (src + hop) % num_points;
            if (dst == src)
               continue;
            bool dup = false;
            for (unsigned j = 0; j < final_count; ++j)
            {
               if (final_nei[j] == dst)
               {
                  dup = true;
                  break;
               }
            }
            if (!dup)
               final_nei[final_count++] = dst;
         }
      }
   }
   __syncthreads();

   for (unsigned i = tid; i < final_degree; i += blockDim.x)
      graph[static_cast<size_t>(src) * k + 1 + i] = i < final_count ? final_nei[i] : src;
   if (tid == 0)
      graph[static_cast<size_t>(src) * k] = final_count;
}

__global__ void fast_grnnd_light_reverse_prune_kernel(unsigned *graph,
                                                      const unsigned *sampled_reverse,
                                                      const unsigned *sampled_reverse_num,
                                                      unsigned num_points,
                                                      unsigned k,
                                                      unsigned final_degree,
                                                      unsigned head_keep,
                                                      unsigned reverse_cap,
                                                      unsigned reverse_slots,
                                                      unsigned repair_degree)
{
   const unsigned src = blockIdx.x;
   const unsigned tid = threadIdx.x;
   if (src >= num_points)
      return;

   __shared__ unsigned final_nei[FINAL_DEGREE_SIZE];
   __shared__ unsigned final_count;
   if (tid == 0)
   {
      final_count = 0;
      const unsigned degree = min(graph[static_cast<size_t>(src) * k], k > 0 ? k - 1 : 0);
      const unsigned near_limit = min(degree, min(head_keep, final_degree));

      for (unsigned scan = 0; scan < near_limit && final_count < final_degree; ++scan)
      {
         const unsigned dst = graph[static_cast<size_t>(src) * k + 1 + scan];
         if (dst >= num_points || dst == src)
            continue;
         bool dup = false;
         for (unsigned j = 0; j < final_count; ++j)
         {
            if (final_nei[j] == dst)
            {
               dup = true;
               break;
            }
         }
         if (!dup)
            final_nei[final_count++] = dst;
      }

      const unsigned reverse_limit = reverse_cap > 0 ? min(sampled_reverse_num[src], reverse_cap) : 0;
      for (unsigned scan = 0; scan < reverse_limit && final_count < final_degree && scan < reverse_slots; ++scan)
      {
         const unsigned dst = sampled_reverse[static_cast<size_t>(src) * reverse_cap + scan];
         if (dst >= num_points || dst == src)
            continue;
         bool dup = false;
         for (unsigned j = 0; j < final_count; ++j)
         {
            if (final_nei[j] == dst)
            {
               dup = true;
               break;
            }
         }
         if (!dup)
            final_nei[final_count++] = dst;
      }

      if (final_count < final_degree)
      {
         const unsigned remaining = final_degree - final_count;
         const unsigned tail_start = near_limit;
         const unsigned tail_count = degree > tail_start ? degree - tail_start : 0;
         const unsigned step = tail_count > remaining ? (tail_count + remaining - 1) / remaining : 1;
         for (unsigned scan = tail_start; scan < degree && final_count < final_degree; scan += step)
         {
            const unsigned dst = graph[static_cast<size_t>(src) * k + 1 + scan];
            if (dst >= num_points || dst == src)
               continue;
            bool dup = false;
            for (unsigned j = 0; j < final_count; ++j)
            {
               if (final_nei[j] == dst)
               {
                  dup = true;
                  break;
               }
            }
            if (!dup)
               final_nei[final_count++] = dst;
         }
      }

      for (unsigned scan = near_limit; scan < degree && final_count < final_degree; ++scan)
      {
         const unsigned dst = graph[static_cast<size_t>(src) * k + 1 + scan];
         if (dst >= num_points || dst == src)
            continue;
         bool dup = false;
         for (unsigned j = 0; j < final_count; ++j)
         {
            if (final_nei[j] == dst)
            {
               dup = true;
               break;
            }
         }
         if (!dup)
            final_nei[final_count++] = dst;
      }
      if (repair_degree && num_points > 1)
      {
         for (unsigned hop = 1; hop < num_points && final_count < final_degree; ++hop)
         {
            const unsigned dst = (src + hop) % num_points;
            if (dst == src)
               continue;
            bool dup = false;
            for (unsigned j = 0; j < final_count; ++j)
            {
               if (final_nei[j] == dst)
               {
                  dup = true;
                  break;
               }
            }
            if (!dup)
               final_nei[final_count++] = dst;
         }
      }
   }
   __syncthreads();

   for (unsigned i = tid; i < final_degree; i += blockDim.x)
      graph[static_cast<size_t>(src) * k + 1 + i] = i < final_count ? final_nei[i] : src;
   if (tid == 0)
      graph[static_cast<size_t>(src) * k] = final_count;
}

__device__ __forceinline__ float warp_reduce_sum(float value)
{
   for (int offset = 16; offset > 0; offset >>= 1)
      value += __shfl_down_sync(0xffffffff, value, offset);
   return value;
}

__device__ __forceinline__ float block_reduce_sum(float value)
{
   __shared__ float warp_sums[32];
   const int lane = threadIdx.x & 31;
   const int warp = threadIdx.x >> 5;
   value = warp_reduce_sum(value);
   if (lane == 0)
      warp_sums[warp] = value;
   __syncthreads();

   value = threadIdx.x < (blockDim.x + 31) / 32 ? warp_sums[lane] : 0.0f;
   if (warp == 0)
      value = warp_reduce_sum(value);
   return value;
}

__global__ void batched_exact_knn_graph_kernel(const float *data,
                                               const uint32_t *offsets,
                                               const uint32_t *sizes,
                                               const uint32_t *point_group_ids,
                                               const uint32_t *point_local_ids,
                                               uint32_t num_groups,
                                               uint32_t dim,
                                               uint32_t candidate_k,
                                               uint32_t graph_stride,
                                               uint32_t final_degree,
                                               uint32_t head_keep,
                                               uint32_t anchor_tail,
                                               uint32_t anchor_slots,
                                               uint32_t bidir_anchor,
                                               uint32_t *graph,
                                               uint32_t *entry_points)
{
   const uint32_t global_src = blockIdx.x;
   if (global_src >= offsets[num_groups])
      return;

   const uint32_t group_id = point_group_ids ? point_group_ids[global_src] : 0;
   const uint32_t group_begin = offsets[group_id];
   const uint32_t n = sizes[group_id];
   const uint32_t local_src = point_local_ids ? point_local_ids[global_src] : (global_src - group_begin);
   if (local_src >= n)
      return;
   if (entry_points && local_src == 0)
      entry_points[group_id] = 0;

   constexpr uint32_t KMAX = 64;
   __shared__ uint32_t top_idx[KMAX];
   __shared__ float top_dist[KMAX];
   __shared__ float reduced_dist;

   const uint32_t cand_degree = min(KMAX, min(candidate_k > 0 ? candidate_k - 1 : 0, n > 0 ? n - 1 : 0));
   const uint32_t out_degree = min(final_degree, cand_degree);
   if (threadIdx.x < KMAX)
   {
      top_idx[threadIdx.x] = 0xffffffffu;
      top_dist[threadIdx.x] = 3.402823466e+38F;
   }
   __syncthreads();

   const float *src = data + static_cast<size_t>(global_src) * dim;
   for (uint32_t local_dst = 0; local_dst < n; ++local_dst)
   {
      if (local_dst == local_src)
         continue;
      const uint32_t global_dst = group_begin + local_dst;
      const float *dst = data + static_cast<size_t>(global_dst) * dim;
      float acc = 0.0f;
      for (uint32_t d = threadIdx.x; d < dim; d += blockDim.x)
      {
         const float diff = src[d] - dst[d];
         acc += diff * diff;
      }
      acc = block_reduce_sum(acc);
      if (threadIdx.x == 0)
         reduced_dist = acc;
      __syncthreads();

      if (threadIdx.x == 0 && cand_degree > 0 && reduced_dist < top_dist[cand_degree - 1])
      {
         uint32_t pos = cand_degree;
         while (pos > 0 && reduced_dist < top_dist[pos - 1])
         {
            if (pos < cand_degree)
            {
               top_dist[pos] = top_dist[pos - 1];
               top_idx[pos] = top_idx[pos - 1];
            }
            --pos;
         }
         top_dist[pos] = reduced_dist;
         top_idx[pos] = local_dst;
      }
      __syncthreads();
   }

   uint32_t *row = graph + static_cast<size_t>(global_src) * graph_stride;
   if (threadIdx.x == 0)
      row[0] = out_degree;
   if (threadIdx.x == 0 && out_degree > 0)
   {
      const uint32_t near_limit = min(out_degree, min(head_keep, final_degree));
      uint32_t write = 0;
      for (; write < near_limit; ++write)
         row[1 + write] = top_idx[write];

      const uint32_t remaining = final_degree > write ? final_degree - write : 0;
      const uint32_t anchor_limit = anchor_tail ? (anchor_slots == 0 ? remaining : min(anchor_slots, remaining)) : 0;
      if (anchor_limit > 0 && n > 1)
      {
         for (uint32_t t = 0; t < anchor_limit && write < out_degree; ++t)
         {
            const uint32_t hop = max(1u, ((t + 1) * n) / (anchor_limit + 1));
            const bool backward = bidir_anchor && (t & 1u);
            uint32_t candidate = backward ? (local_src + n - (hop % n)) % n : (local_src + hop) % n;
            for (uint32_t retry = 0; retry < n && write < out_degree; ++retry, candidate = backward ? (candidate + n - 1) % n : (candidate + 1) % n)
            {
               if (candidate == local_src)
                  continue;
               bool duplicate = false;
               for (uint32_t prev = 0; prev < write; ++prev)
               {
                  if (row[1 + prev] == candidate)
                  {
                     duplicate = true;
                     break;
                  }
               }
               if (!duplicate)
               {
                  row[1 + write++] = candidate;
                  break;
               }
            }
         }
      }

      const uint32_t tail_start = near_limit;
      const uint32_t tail_count = cand_degree > tail_start ? cand_degree - tail_start : 0;
      const uint32_t step = tail_count > remaining && remaining > 0 ? (tail_count + remaining - 1) / remaining : 1;
      for (uint32_t scan = tail_start; scan < cand_degree && write < out_degree; scan += step)
         row[1 + write++] = top_idx[scan];
      for (uint32_t scan = tail_start; scan < cand_degree && write < out_degree; ++scan)
      {
         uint32_t candidate = top_idx[scan];
         bool duplicate = false;
         for (uint32_t prev = 0; prev < write; ++prev)
         {
            if (row[1 + prev] == candidate)
            {
               duplicate = true;
               break;
            }
         }
         if (!duplicate)
            row[1 + write++] = candidate;
      }
   }
   __syncthreads();
   for (uint32_t j = threadIdx.x + out_degree; j < graph_stride - 1; j += blockDim.x)
      row[1 + j] = local_src;
}

__global__ void batched_exact_knn_graph_warp_kernel(const float *data,
                                                    const uint32_t *offsets,
                                                    const uint32_t *sizes,
                                                    const uint32_t *point_group_ids,
                                                    const uint32_t *point_local_ids,
                                                    uint32_t num_groups,
                                                    uint32_t total_points,
                                                    uint32_t dim,
                                                    uint32_t candidate_k,
                                                    uint32_t graph_stride,
                                                    uint32_t final_degree,
                                                    uint32_t head_keep,
                                                    uint32_t anchor_tail,
                                                    uint32_t anchor_slots,
                                                    uint32_t bidir_anchor,
                                                    uint32_t *graph,
                                                    uint32_t *entry_points)
{
   constexpr uint32_t KMAX = 64;
   constexpr uint32_t WARPS_PER_BLOCK = 4;
   __shared__ uint32_t top_idx[WARPS_PER_BLOCK][KMAX];
   __shared__ float top_dist[WARPS_PER_BLOCK][KMAX];

   const uint32_t lane = threadIdx.x & 31u;
   const uint32_t warp = threadIdx.x >> 5u;
   const uint32_t global_src = blockIdx.x * WARPS_PER_BLOCK + warp;
   if (warp >= WARPS_PER_BLOCK || global_src >= total_points)
      return;

   const uint32_t group_id = point_group_ids ? point_group_ids[global_src] : 0;
   if (group_id >= num_groups)
      return;
   const uint32_t group_begin = offsets[group_id];
   const uint32_t n = sizes[group_id];
   const uint32_t local_src = point_local_ids ? point_local_ids[global_src] : (global_src - group_begin);
   if (local_src >= n)
      return;
   if (entry_points && local_src == 0 && lane == 0)
      entry_points[group_id] = 0;

   const uint32_t cand_degree = min(KMAX, min(candidate_k > 0 ? candidate_k - 1 : 0, n > 0 ? n - 1 : 0));
   const uint32_t out_degree = min(final_degree, cand_degree);
   for (uint32_t j = lane; j < KMAX; j += 32)
   {
      top_idx[warp][j] = 0xffffffffu;
      top_dist[warp][j] = 3.402823466e+38F;
   }
   __syncwarp();

   const float *src = data + static_cast<size_t>(global_src) * dim;
   for (uint32_t local_dst = 0; local_dst < n; ++local_dst)
   {
      if (local_dst == local_src)
         continue;
      const uint32_t global_dst = group_begin + local_dst;
      const float *dst = data + static_cast<size_t>(global_dst) * dim;
      float acc = 0.0f;
      for (uint32_t d = lane; d < dim; d += 32)
      {
         const float diff = src[d] - dst[d];
         acc += diff * diff;
      }
      for (uint32_t offset = 16; offset > 0; offset >>= 1)
         acc += __shfl_down_sync(0xffffffffu, acc, offset);

      if (lane == 0 && cand_degree > 0 && acc < top_dist[warp][cand_degree - 1])
      {
         uint32_t pos = cand_degree;
         while (pos > 0 && acc < top_dist[warp][pos - 1])
         {
            if (pos < cand_degree)
            {
               top_dist[warp][pos] = top_dist[warp][pos - 1];
               top_idx[warp][pos] = top_idx[warp][pos - 1];
            }
            --pos;
         }
         top_dist[warp][pos] = acc;
         top_idx[warp][pos] = local_dst;
      }
   }

   if (lane != 0)
      return;

   uint32_t *row = graph + static_cast<size_t>(global_src) * graph_stride;
   row[0] = out_degree;
   if (out_degree == 0)
   {
      for (uint32_t j = 0; j < graph_stride - 1; ++j)
         row[1 + j] = local_src;
      return;
   }

   const uint32_t near_limit = min(out_degree, min(head_keep, final_degree));
   uint32_t write = 0;
   for (; write < near_limit; ++write)
      row[1 + write] = top_idx[warp][write];

   const uint32_t remaining = final_degree > write ? final_degree - write : 0;
   const uint32_t anchor_limit = anchor_tail ? (anchor_slots == 0 ? remaining : min(anchor_slots, remaining)) : 0;
   if (anchor_limit > 0 && n > 1)
   {
      for (uint32_t t = 0; t < anchor_limit && write < out_degree; ++t)
      {
         const uint32_t hop = max(1u, ((t + 1) * n) / (anchor_limit + 1));
         const bool backward = bidir_anchor && (t & 1u);
         uint32_t candidate = backward ? (local_src + n - (hop % n)) % n : (local_src + hop) % n;
         for (uint32_t retry = 0; retry < n && write < out_degree; ++retry, candidate = backward ? (candidate + n - 1) % n : (candidate + 1) % n)
         {
            if (candidate == local_src)
               continue;
            bool duplicate = false;
            for (uint32_t prev = 0; prev < write; ++prev)
            {
               if (row[1 + prev] == candidate)
               {
                  duplicate = true;
                  break;
               }
            }
            if (!duplicate)
            {
               row[1 + write++] = candidate;
               break;
            }
         }
      }
   }

   const uint32_t tail_start = near_limit;
   const uint32_t tail_count = cand_degree > tail_start ? cand_degree - tail_start : 0;
   const uint32_t step = tail_count > remaining && remaining > 0 ? (tail_count + remaining - 1) / remaining : 1;
   for (uint32_t scan = tail_start; scan < cand_degree && write < out_degree; scan += step)
      row[1 + write++] = top_idx[warp][scan];
   for (uint32_t scan = tail_start; scan < cand_degree && write < out_degree; ++scan)
   {
      uint32_t candidate = top_idx[warp][scan];
      bool duplicate = false;
      for (uint32_t prev = 0; prev < write; ++prev)
      {
         if (row[1 + prev] == candidate)
         {
            duplicate = true;
            break;
         }
      }
      if (!duplicate)
         row[1 + write++] = candidate;
   }
   for (uint32_t j = out_degree; j < graph_stride - 1; ++j)
      row[1 + j] = local_src;
}

__global__ void fill_batched_exact_point_lookup_kernel(const uint32_t *offsets,
                                                       const uint32_t *sizes,
                                                       uint32_t num_groups,
                                                       uint32_t *point_group_ids,
                                                       uint32_t *point_local_ids)
{
   const uint32_t group_id = blockIdx.x;
   if (group_id >= num_groups)
      return;
   const uint32_t begin = offsets[group_id];
   const uint32_t n = sizes[group_id];
   for (uint32_t local = threadIdx.x; local < n; local += blockDim.x)
   {
      const uint32_t global = begin + local;
      point_group_ids[global] = group_id;
      point_local_ids[global] = local;
   }
}

__global__ void build_exact_reverse_local_kernel(const uint32_t *graph,
                                                 const uint32_t *offsets,
                                                 const uint32_t *sizes,
                                                 const uint32_t *point_group_ids,
                                                 const uint32_t *point_local_ids,
                                                 uint32_t num_groups,
                                                 uint32_t total_points,
                                                 uint32_t graph_stride,
                                                 uint32_t forward_cap,
                                                 uint32_t reverse_cap,
                                                 uint32_t *reverse_ids,
                                                 uint32_t *reverse_counts)
{
   const uint32_t global_src = blockIdx.x * blockDim.x + threadIdx.x;
   if (global_src >= total_points)
      return;
   const uint32_t group_id = point_group_ids[global_src];
   if (group_id >= num_groups)
      return;
   const uint32_t group_begin = offsets[group_id];
   const uint32_t n = sizes[group_id];
   const uint32_t local_src = point_local_ids[global_src];
   if (local_src >= n)
      return;

   const uint32_t *row = graph + static_cast<size_t>(global_src) * graph_stride;
   const uint32_t degree = min(row[0], graph_stride > 0 ? graph_stride - 1 : 0);
   const uint32_t limit = min(degree, forward_cap);
   for (uint32_t i = 0; i < limit; ++i)
   {
      const uint32_t local_dst = row[1 + i];
      if (local_dst >= n || local_dst == local_src)
         continue;
      const uint32_t global_dst = group_begin + local_dst;
      const uint32_t pos = atomicAdd(reverse_counts + global_dst, 1u);
      if (pos < reverse_cap)
         reverse_ids[static_cast<size_t>(global_dst) * reverse_cap + pos] = local_src;
   }
}

__global__ void apply_exact_reverse_slots_kernel(uint32_t *graph,
                                                 const uint32_t *sizes,
                                                 const uint32_t *point_group_ids,
                                                 const uint32_t *point_local_ids,
                                                 const uint32_t *reverse_ids,
                                                 const uint32_t *reverse_counts,
                                                 uint32_t total_points,
                                                 uint32_t graph_stride,
                                                 uint32_t final_degree,
                                                 uint32_t reverse_cap,
                                                 uint32_t reverse_slots)
{
   const uint32_t global_src = blockIdx.x;
   if (global_src >= total_points || reverse_slots == 0 || reverse_cap == 0)
      return;
   const uint32_t group_id = point_group_ids[global_src];
   const uint32_t n = sizes[group_id];
   const uint32_t local_src = point_local_ids[global_src];
   uint32_t *row = graph + static_cast<size_t>(global_src) * graph_stride;
   const uint32_t degree = min(row[0], graph_stride > 0 ? graph_stride - 1 : 0);
   if (degree == 0)
      return;

   uint32_t inserted = 0;
   const uint32_t rev_count = min(reverse_counts[global_src], reverse_cap);
   for (uint32_t scan = 0; scan < rev_count && inserted < reverse_slots; ++scan)
   {
      const uint32_t candidate = reverse_ids[static_cast<size_t>(global_src) * reverse_cap + scan];
      if (candidate >= n || candidate == local_src)
         continue;
      bool duplicate = false;
      for (uint32_t j = 0; j < degree; ++j)
      {
         if (row[1 + j] == candidate)
         {
            duplicate = true;
            break;
         }
      }
      if (duplicate)
         continue;

      const uint32_t replace_pos = degree - 1 - inserted;
      row[1 + replace_pos] = candidate;
      ++inserted;
   }
}

TagoreBatchBuildResult build_fast_exact_cuda_batch(const std::vector<TagoreGroupRequest> &groups,
                                                   uint32_t dim,
                                                   uint32_t k,
                                                   uint32_t final_degree,
                                                   const TagoreFastExactBatchConfig &config)
{
   TagoreBatchBuildResult batch;
   batch.groups.resize(groups.size());

   uint64_t total_points_u64 = 0;
   for (const auto &g : groups)
      total_points_u64 += g.num_points;
   if (total_points_u64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()))
      throw std::runtime_error("Fast exact CUDA batch supports at most uint32_t total points.");
   const uint32_t total_points = static_cast<uint32_t>(total_points_u64);
   const uint32_t head_keep = config.head_keep;
   const uint32_t anchor_tail = config.anchor_tail;
   const uint32_t anchor_slots = config.anchor_slots;
   const uint32_t bidir_anchor = config.bidir_anchor;
   const uint32_t exact_reverse_cap = config.reverse_cap;
   const uint32_t exact_reverse_slots = config.reverse_slots;
   const uint32_t exact_reverse_forward_cap = config.reverse_forward_cap;
   const bool pinned_host = config.pinned_host;
   const bool device_lookup = config.device_lookup;
   const bool direct_h2d_req = config.direct_h2d_requested;
   const uint32_t direct_h2d_max_runs = config.direct_h2d_max_runs;
   const uint32_t graph_stride = config.graph_stride;
   const bool use_warp_kernel = config.use_warp_kernel;

   std::vector<uint32_t> offsets(groups.size() + 1, 0);
   std::vector<uint32_t> sizes(groups.size(), 0);
   std::vector<uint32_t> point_group_ids;
   std::vector<uint32_t> point_local_ids;
   if (!device_lookup)
   {
      point_group_ids.resize(total_points);
      point_local_ids.resize(total_points);
   }
   const auto pack_start = std::chrono::high_resolution_clock::now();
   struct DirectRun
   {
      const float *src = nullptr;
      uint32_t dst_offset = 0;
      uint32_t points = 0;
   };
   std::vector<DirectRun> direct_runs;
   direct_runs.reserve(groups.size());

   for (size_t gi = 0; gi < groups.size(); ++gi)
   {
      sizes[gi] = groups[gi].num_points;
      offsets[gi + 1] = offsets[gi] + groups[gi].num_points;
      if (direct_h2d_req && groups[gi].data && groups[gi].num_points > 0)
      {
         if (!direct_runs.empty())
         {
            DirectRun &run = direct_runs.back();
            const float *expected = run.src + static_cast<size_t>(run.points) * dim;
            if (groups[gi].data == expected && offsets[gi] == run.dst_offset + run.points)
            {
               run.points += groups[gi].num_points;
            }
            else
            {
               direct_runs.push_back({groups[gi].data, offsets[gi], groups[gi].num_points});
            }
         }
         else
         {
            direct_runs.push_back({groups[gi].data, offsets[gi], groups[gi].num_points});
         }
      }
   }
   const bool use_direct_h2d =
       direct_h2d_req && !direct_runs.empty() && direct_runs.size() <= direct_h2d_max_runs;

   float *packed_pinned = nullptr;
   std::vector<float> packed_fallback;
   float *packed_data = nullptr;
   const size_t packed_values = static_cast<size_t>(total_points) * dim;
   if (!use_direct_h2d && pinned_host)
   {
      cudaError_t pin_err = cudaMallocHost(&packed_pinned, packed_values * sizeof(float));
      if (pin_err == cudaSuccess)
      {
         packed_data = packed_pinned;
      }
      else
      {
         cudaGetLastError();
         packed_fallback.resize(packed_values);
         packed_data = packed_fallback.data();
      }
   }
   else if (!use_direct_h2d)
   {
      packed_fallback.resize(packed_values);
      packed_data = packed_fallback.data();
   }

   for (size_t gi = 0; gi < groups.size(); ++gi)
   {
      if (!use_direct_h2d)
      {
         std::memcpy(packed_data + static_cast<size_t>(offsets[gi]) * dim,
                     groups[gi].data,
                     static_cast<size_t>(groups[gi].num_points) * dim * sizeof(float));
      }
      if (!device_lookup)
      {
         for (uint32_t local = 0; local < groups[gi].num_points; ++local)
         {
            const uint32_t global = offsets[gi] + local;
            point_group_ids[global] = static_cast<uint32_t>(gi);
            point_local_ids[global] = local;
         }
      }
      batch.groups[gi].graph_stride = graph_stride;
      batch.groups[gi].entry_point = 0;
   }
   batch.packed_offsets = offsets;
   batch.packed_graph_stride = graph_stride;
   batch.batch_pack_ms = elapsed_ms(pack_start);
   std::cout << "[FastExact] direct_h2d=" << (use_direct_h2d ? 1 : 0)
             << " runs=" << direct_runs.size()
             << " max_runs=" << direct_h2d_max_runs
             << " device_lookup=" << (device_lookup ? 1 : 0)
             << " warp_kernel=" << (use_warp_kernel ? 1 : 0)
             << " pack_ms=" << batch.batch_pack_ms
             << std::endl;

   float *data_dev = nullptr;
   uint32_t *offsets_dev = nullptr;
   uint32_t *sizes_dev = nullptr;
   uint32_t *point_group_ids_dev = nullptr;
   uint32_t *point_local_ids_dev = nullptr;
   uint32_t *graph_dev = nullptr;
   uint32_t *entry_dev = nullptr;
   uint32_t *exact_reverse_ids_dev = nullptr;
   uint32_t *exact_reverse_counts_dev = nullptr;
   uint32_t *graph_host_pinned = nullptr;
   uint32_t *entries_pinned = nullptr;
   cudaStream_t stream = nullptr;

   try
   {
      ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate exact batch");
      const auto alloc_start = std::chrono::high_resolution_clock::now();
      ck(cudaMallocAsync(&data_dev, packed_values * sizeof(float), stream), "cudaMallocAsync exact data");
      ck(cudaMallocAsync(&offsets_dev, offsets.size() * sizeof(uint32_t), stream), "cudaMallocAsync exact offsets");
      ck(cudaMallocAsync(&sizes_dev, sizes.size() * sizeof(uint32_t), stream), "cudaMallocAsync exact sizes");
      if (device_lookup)
      {
         ck(cudaMallocAsync(&point_group_ids_dev, static_cast<size_t>(total_points) * sizeof(uint32_t), stream),
            "cudaMallocAsync exact point group ids");
         ck(cudaMallocAsync(&point_local_ids_dev, static_cast<size_t>(total_points) * sizeof(uint32_t), stream),
            "cudaMallocAsync exact point local ids");
      }
      else
      {
         ck(cudaMallocAsync(&point_group_ids_dev, point_group_ids.size() * sizeof(uint32_t), stream),
            "cudaMallocAsync exact point group ids");
         ck(cudaMallocAsync(&point_local_ids_dev, point_local_ids.size() * sizeof(uint32_t), stream),
            "cudaMallocAsync exact point local ids");
      }
      ck(cudaMallocAsync(&graph_dev, static_cast<size_t>(total_points) * graph_stride * sizeof(uint32_t), stream),
         "cudaMallocAsync exact graph");
      ck(cudaMallocAsync(&entry_dev, groups.size() * sizeof(uint32_t), stream), "cudaMallocAsync exact entry");
      if (exact_reverse_cap > 0 && exact_reverse_slots > 0)
      {
         ck(cudaMallocAsync(&exact_reverse_ids_dev, static_cast<size_t>(total_points) * exact_reverse_cap * sizeof(uint32_t), stream),
            "cudaMallocAsync exact reverse ids");
         ck(cudaMallocAsync(&exact_reverse_counts_dev, static_cast<size_t>(total_points) * sizeof(uint32_t), stream),
            "cudaMallocAsync exact reverse counts");
      }
      if (pinned_host)
      {
         const size_t graph_host_bytes = static_cast<size_t>(total_points) * graph_stride * sizeof(uint32_t);
         cudaError_t graph_pin_err = cudaMallocHost(&graph_host_pinned, graph_host_bytes);
         cudaError_t entry_pin_err = cudaMallocHost(&entries_pinned, groups.size() * sizeof(uint32_t));
         if (graph_pin_err != cudaSuccess || entry_pin_err != cudaSuccess)
         {
            cudaGetLastError();
            if (graph_host_pinned)
            {
               cudaFreeHost(graph_host_pinned);
               graph_host_pinned = nullptr;
            }
            if (entries_pinned)
            {
               cudaFreeHost(entries_pinned);
               entries_pinned = nullptr;
            }
         }
      }
      ck(cudaStreamSynchronize(stream), "cudaMallocAsync exact sync");
      batch.batch_alloc_ms = elapsed_ms(alloc_start);

      const auto h2d_start = std::chrono::high_resolution_clock::now();
      if (use_direct_h2d)
      {
         for (const DirectRun &run : direct_runs)
         {
            ck(cudaMemcpyAsync(data_dev + static_cast<size_t>(run.dst_offset) * dim,
                               run.src,
                               static_cast<size_t>(run.points) * dim * sizeof(float),
                               cudaMemcpyHostToDevice,
                               stream),
               "cudaMemcpyAsync exact direct data");
         }
      }
      else
      {
         ck(cudaMemcpyAsync(data_dev, packed_data, packed_values * sizeof(float), cudaMemcpyHostToDevice, stream),
            "cudaMemcpyAsync exact data");
      }
      ck(cudaMemcpyAsync(offsets_dev, offsets.data(), offsets.size() * sizeof(uint32_t), cudaMemcpyHostToDevice, stream),
         "cudaMemcpyAsync exact offsets");
      ck(cudaMemcpyAsync(sizes_dev, sizes.data(), sizes.size() * sizeof(uint32_t), cudaMemcpyHostToDevice, stream),
         "cudaMemcpyAsync exact sizes");
      if (!device_lookup)
      {
         ck(cudaMemcpyAsync(point_group_ids_dev, point_group_ids.data(), point_group_ids.size() * sizeof(uint32_t),
                            cudaMemcpyHostToDevice, stream),
            "cudaMemcpyAsync exact point group ids");
         ck(cudaMemcpyAsync(point_local_ids_dev, point_local_ids.data(), point_local_ids.size() * sizeof(uint32_t),
                            cudaMemcpyHostToDevice, stream),
            "cudaMemcpyAsync exact point local ids");
      }
      ck(cudaStreamSynchronize(stream), "cudaMemcpyAsync exact h2d sync");
      const double h2d_ms = elapsed_ms(h2d_start);

      double lookup_ms = 0.0;
      if (device_lookup)
      {
         const auto lookup_start = std::chrono::high_resolution_clock::now();
         fill_batched_exact_point_lookup_kernel<<<static_cast<uint32_t>(groups.size()), 256, 0, stream>>>(
             offsets_dev, sizes_dev, static_cast<uint32_t>(groups.size()),
             point_group_ids_dev, point_local_ids_dev);
         ck(cudaGetLastError(), "fill exact point lookup launch");
         ck(cudaStreamSynchronize(stream), "fill exact point lookup sync");
         lookup_ms = elapsed_ms(lookup_start);
      }

      const auto gnn_start = std::chrono::high_resolution_clock::now();
      if (use_warp_kernel)
      {
         constexpr uint32_t warps_per_block = 4;
         const uint32_t blocks = (total_points + warps_per_block - 1) / warps_per_block;
         batched_exact_knn_graph_warp_kernel<<<blocks, warps_per_block * 32, 0, stream>>>(
             data_dev, offsets_dev, sizes_dev, point_group_ids_dev, point_local_ids_dev,
             static_cast<uint32_t>(groups.size()), total_points, dim, k, graph_stride, final_degree,
             head_keep, anchor_tail, anchor_slots, bidir_anchor, graph_dev, entry_dev);
      }
      else
      {
         batched_exact_knn_graph_kernel<<<total_points, 128, 0, stream>>>(
             data_dev, offsets_dev, sizes_dev, point_group_ids_dev, point_local_ids_dev,
             static_cast<uint32_t>(groups.size()), dim, k, graph_stride, final_degree,
             head_keep, anchor_tail, anchor_slots, bidir_anchor, graph_dev, entry_dev);
      }
      ck(cudaGetLastError(), "batched exact graph launch");
      if (exact_reverse_cap > 0 && exact_reverse_slots > 0)
      {
         ck(cudaMemsetAsync(exact_reverse_counts_dev, 0, static_cast<size_t>(total_points) * sizeof(uint32_t), stream),
            "cudaMemsetAsync exact reverse counts");
         const uint32_t reverse_threads = 256;
         const uint32_t reverse_blocks = (total_points + reverse_threads - 1) / reverse_threads;
         build_exact_reverse_local_kernel<<<reverse_blocks, reverse_threads, 0, stream>>>(
             graph_dev, offsets_dev, sizes_dev, point_group_ids_dev, point_local_ids_dev,
             static_cast<uint32_t>(groups.size()), total_points, graph_stride,
             exact_reverse_forward_cap, exact_reverse_cap, exact_reverse_ids_dev, exact_reverse_counts_dev);
         ck(cudaGetLastError(), "build exact reverse launch");
         apply_exact_reverse_slots_kernel<<<total_points, 32, 0, stream>>>(
             graph_dev, sizes_dev, point_group_ids_dev, point_local_ids_dev,
             exact_reverse_ids_dev, exact_reverse_counts_dev, total_points, graph_stride, final_degree,
             exact_reverse_cap, exact_reverse_slots);
         ck(cudaGetLastError(), "apply exact reverse slots launch");
      }
      ck(cudaStreamSynchronize(stream), "batched exact graph synchronize");
      const double gnn_ms = elapsed_ms(gnn_start);

      std::vector<uint32_t> graph_host_fallback;
      std::vector<uint32_t> entries_fallback;
      uint32_t *graph_host = graph_host_pinned;
      uint32_t *entries = entries_pinned;
      if (!graph_host)
      {
         graph_host_fallback.resize(static_cast<size_t>(total_points) * graph_stride);
         graph_host = graph_host_fallback.data();
      }
      if (!entries)
      {
         entries_fallback.resize(groups.size(), 0);
         entries = entries_fallback.data();
      }
      const auto d2h_start = std::chrono::high_resolution_clock::now();
      ck(cudaMemcpyAsync(graph_host, graph_dev, static_cast<size_t>(total_points) * graph_stride * sizeof(uint32_t),
                         cudaMemcpyDeviceToHost, stream),
         "cudaMemcpyAsync exact graph D2H");
      ck(cudaMemcpyAsync(entries, entry_dev, groups.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream),
         "cudaMemcpyAsync exact entry D2H");
      ck(cudaStreamSynchronize(stream), "cudaMemcpyAsync exact d2h sync");
      const double d2h_ms = elapsed_ms(d2h_start);

      for (size_t gi = 0; gi < groups.size(); ++gi)
      {
         auto &result = batch.groups[gi];
         result.h2d_ms = h2d_ms * (static_cast<double>(groups[gi].num_points) / std::max<uint32_t>(1, total_points));
         result.memset_ms = lookup_ms * (static_cast<double>(groups[gi].num_points) / std::max<uint32_t>(1, total_points));
         result.gnn_ms = gnn_ms * (static_cast<double>(groups[gi].num_points) / std::max<uint32_t>(1, total_points));
         result.d2h_ms = d2h_ms * (static_cast<double>(groups[gi].num_points) / std::max<uint32_t>(1, total_points));
         result.entry_point = entries[gi];
      }
      if (graph_host_pinned)
      {
         batch.packed_graph.assign(graph_host, graph_host + static_cast<size_t>(total_points) * graph_stride);
      }
      else
      {
         batch.packed_graph = std::move(graph_host_fallback);
      }
   }
   catch (...)
   {
      cudaFreeAsync(data_dev, stream);
      cudaFreeAsync(offsets_dev, stream);
      cudaFreeAsync(sizes_dev, stream);
      cudaFreeAsync(point_group_ids_dev, stream);
      cudaFreeAsync(point_local_ids_dev, stream);
      cudaFreeAsync(graph_dev, stream);
      cudaFreeAsync(entry_dev, stream);
      if (exact_reverse_ids_dev)
         cudaFreeAsync(exact_reverse_ids_dev, stream);
      if (exact_reverse_counts_dev)
         cudaFreeAsync(exact_reverse_counts_dev, stream);
      cudaStreamSynchronize(stream);
      if (packed_pinned)
         cudaFreeHost(packed_pinned);
      if (graph_host_pinned)
         cudaFreeHost(graph_host_pinned);
      if (entries_pinned)
         cudaFreeHost(entries_pinned);
      if (stream)
         cudaStreamDestroy(stream);
      throw;
   }

   const auto free_start = std::chrono::high_resolution_clock::now();
   ck(cudaFreeAsync(data_dev, stream), "cudaFreeAsync exact data");
   ck(cudaFreeAsync(offsets_dev, stream), "cudaFreeAsync exact offsets");
   ck(cudaFreeAsync(sizes_dev, stream), "cudaFreeAsync exact sizes");
   ck(cudaFreeAsync(point_group_ids_dev, stream), "cudaFreeAsync exact point group ids");
   ck(cudaFreeAsync(point_local_ids_dev, stream), "cudaFreeAsync exact point local ids");
   ck(cudaFreeAsync(graph_dev, stream), "cudaFreeAsync exact graph");
   ck(cudaFreeAsync(entry_dev, stream), "cudaFreeAsync exact entry");
   if (exact_reverse_ids_dev)
      ck(cudaFreeAsync(exact_reverse_ids_dev, stream), "cudaFreeAsync exact reverse ids");
   if (exact_reverse_counts_dev)
      ck(cudaFreeAsync(exact_reverse_counts_dev, stream), "cudaFreeAsync exact reverse counts");
   ck(cudaStreamSynchronize(stream), "cudaFreeAsync exact sync");
   batch.batch_free_ms = elapsed_ms(free_start);
   ck(cudaStreamDestroy(stream), "cudaStreamDestroy exact batch");
   if (packed_pinned)
      ck(cudaFreeHost(packed_pinned), "cudaFreeHost exact packed");
   if (graph_host_pinned)
      ck(cudaFreeHost(graph_host_pinned), "cudaFreeHost exact graph host");
   if (entries_pinned)
      ck(cudaFreeHost(entries_pinned), "cudaFreeHost exact entries");

   return batch;
}

void launch_grnnd_like_refine(unsigned *graph_dev,
                              const half *data_dev,
                              uint32_t num_points,
                              uint32_t dim,
                              uint32_t k,
                              uint32_t final_degree,
                              float alpha,
                              double &refine_ms)
{
   const uint32_t outgoing_sample_limit = std::min<uint32_t>(k > 0 ? k - 1 : 1,
                                                             std::max<uint32_t>(1, static_cast<uint32_t>(std::ceil(0.6f * final_degree))));
   const uint32_t reverse_cap = outgoing_sample_limit;
   const float refine_alpha = alpha;
   unsigned *sampled_reverse = nullptr;
   unsigned *sampled_reverse_num = nullptr;
   ck(cudaMalloc(&sampled_reverse, static_cast<size_t>(num_points) * reverse_cap * sizeof(unsigned)), "cudaMalloc sampled_reverse");
   ck(cudaMalloc(&sampled_reverse_num, static_cast<size_t>(num_points) * sizeof(unsigned)), "cudaMalloc sampled_reverse_num");
   ck(cudaMemset(sampled_reverse_num, 0, static_cast<size_t>(num_points) * sizeof(unsigned)), "cudaMemset sampled_reverse_num");

   const auto start = std::chrono::high_resolution_clock::now();
   const int threads = 256;
   const int blocks = static_cast<int>((num_points + threads - 1) / threads);
   build_sampled_reverse_kernel<<<blocks, threads>>>(graph_dev, sampled_reverse, sampled_reverse_num,
                                                     num_points, k, outgoing_sample_limit, reverse_cap);
   grnnd_like_refine_kernel<<<num_points, 32>>>(graph_dev, data_dev, sampled_reverse, sampled_reverse_num,
                                                num_points, dim, k, final_degree, reverse_cap, refine_alpha);
   ck(cudaDeviceSynchronize(), "GRNND-like refine synchronize");
   refine_ms = elapsed_ms(start);
   cudaFree(sampled_reverse);
   cudaFree(sampled_reverse_num);
}

void launch_fast_grnnd_prune_impl(unsigned *graph_dev,
                                  const half *data_dev,
                                  const float *nei_distance,
                                  uint32_t num_points,
                                  uint32_t dim,
                                  uint32_t k,
                                  uint32_t final_degree,
                                  float alpha,
                                  cudaStream_t stream,
                                  unsigned *sampled_reverse_workspace,
                                  unsigned *sampled_reverse_num_workspace,
                                  uint32_t sampled_reverse_workspace_cap,
                                  bool synchronize,
                                  const TagoreFastGrnndPruneConfig &config,
                                  double &prune_ms)
{
   const uint32_t light_threshold = config.light_threshold;
   if (light_threshold > 0 && num_points <= light_threshold)
   {
      const uint32_t head_keep = config.light_head_keep;
      const uint32_t reverse_cap = config.light_reverse_cap;
      const uint32_t repair_degree = config.repair_degree;
      const auto start = std::chrono::high_resolution_clock::now();
      if (reverse_cap > 0)
      {
         const uint32_t reverse_slots = config.light_reverse_slots;
         const uint32_t forward_cap = config.light_reverse_forward_cap;
         const bool own_workspace = !sampled_reverse_workspace || !sampled_reverse_num_workspace ||
                                    sampled_reverse_workspace_cap < reverse_cap;
         unsigned *sampled_reverse = sampled_reverse_workspace;
         unsigned *sampled_reverse_num = sampled_reverse_num_workspace;
         if (own_workspace)
         {
            ck(cudaMalloc(&sampled_reverse, static_cast<size_t>(num_points) * reverse_cap * sizeof(unsigned)),
               "cudaMalloc light sampled_reverse");
            ck(cudaMalloc(&sampled_reverse_num, static_cast<size_t>(num_points) * sizeof(unsigned)),
               "cudaMalloc light sampled_reverse_num");
         }
         ck(cudaMemsetAsync(sampled_reverse_num, 0, static_cast<size_t>(num_points) * sizeof(unsigned), stream),
            "cudaMemsetAsync light sampled_reverse_num");
         const int threads = 256;
         const int blocks = static_cast<int>((num_points + threads - 1) / threads);
         build_sampled_reverse_kernel<<<blocks, threads, 0, stream>>>(graph_dev, sampled_reverse, sampled_reverse_num,
                                                                      num_points, k, forward_cap, reverse_cap);
         fast_grnnd_light_reverse_prune_kernel<<<num_points, 32, 0, stream>>>(graph_dev, sampled_reverse, sampled_reverse_num,
                                                                              num_points, k, final_degree, head_keep,
                                                                              reverse_cap, reverse_slots, repair_degree);
         ck(cudaGetLastError(), "fast GRNND light reverse prune launch");
         if (synchronize)
            ck(cudaStreamSynchronize(stream), "fast GRNND light reverse prune synchronize");
         if (own_workspace)
         {
            cudaFree(sampled_reverse);
            cudaFree(sampled_reverse_num);
         }
      }
      else
      {
         fast_grnnd_light_prune_kernel<<<num_points, 32, 0, stream>>>(graph_dev, num_points, k, final_degree,
                                                                      head_keep, repair_degree);
         ck(cudaGetLastError(), "fast GRNND light prune launch");
         if (synchronize)
            ck(cudaStreamSynchronize(stream), "fast GRNND light prune synchronize");
      }
      prune_ms = elapsed_ms(start);
      return;
   }

   const uint32_t forward_cap = config.forward_cap;
   const uint32_t reverse_cap = config.reverse_cap;
   const bool own_workspace = reverse_cap > 0 &&
                              (!sampled_reverse_workspace || !sampled_reverse_num_workspace ||
                               sampled_reverse_workspace_cap < reverse_cap);
   unsigned *sampled_reverse = sampled_reverse_workspace;
   unsigned *sampled_reverse_num = sampled_reverse_num_workspace;
   if (reverse_cap > 0 && own_workspace)
   {
      ck(cudaMalloc(&sampled_reverse, static_cast<size_t>(num_points) * reverse_cap * sizeof(unsigned)),
         "cudaMalloc fast sampled_reverse");
      ck(cudaMalloc(&sampled_reverse_num, static_cast<size_t>(num_points) * sizeof(unsigned)),
         "cudaMalloc fast sampled_reverse_num");
   }
   if (reverse_cap > 0)
      ck(cudaMemsetAsync(sampled_reverse_num, 0, static_cast<size_t>(num_points) * sizeof(unsigned), stream),
         "cudaMemsetAsync fast sampled_reverse_num");

   const auto start = std::chrono::high_resolution_clock::now();
   const int threads = 256;
   const int blocks = static_cast<int>((num_points + threads - 1) / threads);
   if (reverse_cap > 0)
      build_sampled_reverse_kernel<<<blocks, threads, 0, stream>>>(graph_dev, sampled_reverse, sampled_reverse_num,
                                                                   num_points, k, forward_cap, reverse_cap);
   fast_grnnd_prune_kernel<<<num_points, 32, 0, stream>>>(graph_dev, data_dev, nei_distance,
                                                          sampled_reverse, sampled_reverse_num,
                                                          num_points, dim, k, final_degree,
                                                          forward_cap, reverse_cap, alpha);
   ck(cudaGetLastError(), "fast GRNND prune launch");
   if (synchronize)
      ck(cudaStreamSynchronize(stream), "fast GRNND prune synchronize");
   prune_ms = elapsed_ms(start);
   if (own_workspace)
   {
      cudaFree(sampled_reverse);
      cudaFree(sampled_reverse_num);
   }
}

[[maybe_unused]] void launch_fast_grnnd_prune(unsigned *graph_dev,
                                              const half *data_dev,
                                              const float *nei_distance,
                                              uint32_t num_points,
                                              uint32_t dim,
                                              uint32_t k,
                                              uint32_t final_degree,
                                              float alpha,
                                              double &prune_ms)
{
   const TagoreCudaRuntimeConfig config = make_tagore_cuda_runtime_config(k, final_degree, false);
   launch_fast_grnnd_prune_impl(graph_dev, data_dev, nei_distance, num_points, dim, k, final_degree, alpha,
                                0, nullptr, nullptr, 0, true, config.fast_prune, prune_ms);
}

} // namespace

TagoreBuildResult build_tagore_vamana_cuda(const float *data,
                                           uint32_t num_points,
                                           uint32_t dim,
                                           uint32_t k,
                                           uint32_t final_degree,
                                           uint32_t top_m,
                                           uint32_t iterations,
                                           float alpha)
{
   if (!data)
      throw std::runtime_error("TagoreCuda received null data.");
   if (num_points == 0 || dim == 0)
      throw std::runtime_error("TagoreCuda requires non-empty data.");
   if (k == 0 || final_degree == 0 || top_m == 0)
      throw std::runtime_error("TagoreCuda requires positive k/final_degree/top_m.");
   if (k <= final_degree)
      k = final_degree + 1;

   TagoreBuildResult result;
   result.graph.resize(static_cast<size_t>(num_points) * k);
   result.graph_stride = k;

   const float norm_factor = compute_norm_factor(data, num_points, dim);

   float *data_float_dev = nullptr;
   half *data_dev = nullptr;
   half *center_dev = nullptr;
   unsigned *graph_dev = nullptr;
   unsigned *reverse_graph_dev = nullptr;
   float *nei_distance = nullptr;
   float *reverse_distance = nullptr;
   bool *nei_visit = nullptr;
   unsigned *reverse_num = nullptr;
   unsigned *reverse_num_old = nullptr;
   unsigned *new_num_global = nullptr;
   unsigned *old_num_global = nullptr;
   unsigned *hybrid_list = nullptr;
   float *data_power_dev = nullptr;
   unsigned *ep_dev = nullptr;

   try
   {
      auto alloc_start = std::chrono::high_resolution_clock::now();
      ck(cudaMalloc(&data_float_dev, static_cast<size_t>(num_points) * dim * sizeof(float)), "cudaMalloc data_float_dev");
      ck(cudaMalloc(&data_dev, static_cast<size_t>(num_points) * dim * sizeof(half)), "cudaMalloc data_dev");
      ck(cudaMalloc(&center_dev, static_cast<size_t>(dim) * sizeof(half)), "cudaMalloc center_dev");
      ck(cudaMalloc(&graph_dev, static_cast<size_t>(num_points) * k * sizeof(unsigned)), "cudaMalloc graph_dev");
      ck(cudaMalloc(&reverse_graph_dev, static_cast<size_t>(num_points) * RESERVENUM * sizeof(unsigned)),
         "cudaMalloc reverse_graph_dev");
      ck(cudaMalloc(&nei_distance, static_cast<size_t>(num_points) * k * sizeof(float)), "cudaMalloc nei_distance");
      ck(cudaMalloc(&reverse_distance, static_cast<size_t>(num_points) * RESERVENUM * sizeof(float)),
         "cudaMalloc reverse_distance");
      ck(cudaMalloc(&nei_visit, static_cast<size_t>(num_points) * k * sizeof(bool)), "cudaMalloc nei_visit");
      ck(cudaMalloc(&reverse_num, static_cast<size_t>(num_points) * sizeof(unsigned)), "cudaMalloc reverse_num");
      ck(cudaMalloc(&reverse_num_old, static_cast<size_t>(num_points) * sizeof(unsigned)), "cudaMalloc reverse_num_old");
      ck(cudaMalloc(&new_num_global, static_cast<size_t>(num_points) * sizeof(unsigned)), "cudaMalloc new_num_global");
      ck(cudaMalloc(&old_num_global, static_cast<size_t>(num_points) * sizeof(unsigned)), "cudaMalloc old_num_global");
      ck(cudaMalloc(&hybrid_list, static_cast<size_t>(num_points) * SAMPLE * 4 * sizeof(unsigned)),
         "cudaMalloc hybrid_list");
      ck(cudaMalloc(&data_power_dev, static_cast<size_t>(num_points) * sizeof(float)), "cudaMalloc data_power_dev");
      ck(cudaMalloc(&ep_dev, sizeof(unsigned)), "cudaMalloc ep_dev");
      result.alloc_ms = elapsed_ms(alloc_start);

      auto h2d_start = std::chrono::high_resolution_clock::now();
      ck(cudaMemcpy(data_float_dev, data, static_cast<size_t>(num_points) * dim * sizeof(float),
                    cudaMemcpyHostToDevice),
         "cudaMemcpy data_float_dev");
      result.h2d_ms = elapsed_ms(h2d_start);

      const auto convert_start = std::chrono::high_resolution_clock::now();
      const size_t total_values = static_cast<size_t>(num_points) * dim;
      const int threads = 256;
      const int blocks = static_cast<int>(std::min<size_t>((total_values + threads - 1) / threads, 65535));
      convert_float_to_half_kernel<<<blocks, threads>>>(data_float_dev, data_dev, total_values, norm_factor);
      compute_center_half_kernel<<<dim, threads>>>(data_float_dev, center_dev, num_points, dim, norm_factor);
      ck(cudaDeviceSynchronize(), "float-to-half conversion synchronize");
      result.convert_ms = elapsed_ms(convert_start);

      auto memset_start = std::chrono::high_resolution_clock::now();
      ck(cudaMemset(reverse_num, 0, static_cast<size_t>(num_points) * sizeof(unsigned)), "memset reverse_num");
      ck(cudaMemset(reverse_num_old, 0, static_cast<size_t>(num_points) * sizeof(unsigned)), "memset reverse_num_old");
      ck(cudaMemset(new_num_global, 0, static_cast<size_t>(num_points) * sizeof(unsigned)), "memset new_num_global");
      ck(cudaMemset(old_num_global, 0, static_cast<size_t>(num_points) * sizeof(unsigned)), "memset old_num_global");
      ck(cudaMemset(data_power_dev, 0, static_cast<size_t>(num_points) * sizeof(float)), "memset data_power_dev");
      result.memset_ms = elapsed_ms(memset_start);

      dim3 grid(num_points, 1, 1);
      dim3 block(32, MAX_P / 32, 1);
      dim3 block2(32, 16, 1);
      dim3 block3(32, 3, 1);
      dim3 block_s(32, 4, 1);
      dim3 grid_one(1, 1, 1);

      const auto gnn_start = std::chrono::high_resolution_clock::now();
      initialize_graph<<<num_points, 32>>>(graph_dev, num_points, nei_distance, nei_visit, k);
      cal_power<<<num_points, dim>>>(data_dev, data_power_dev, dim, dim);

      const unsigned all_it = iterations / 2;
      const unsigned all_it2 = iterations / 2;
      for (unsigned it = 0; it < all_it; ++it)
      {
         nn_descent_opt_sample<<<num_points, 32>>>(graph_dev, reverse_graph_dev, nei_visit, reverse_num,
                                                   reverse_num_old, new_num_global, old_num_global, hybrid_list, k);
         nn_descent_opt_reverse_sample<<<num_points, 32>>>(graph_dev, reverse_graph_dev, reverse_num, reverse_num_old,
                                                           new_num_global, old_num_global, hybrid_list, k);
         reset_reverse_new_old_num<<<1000, 1024>>>(reverse_num, reverse_num_old, num_points);
         nn_descent_opt_cal<<<grid, block2>>>(graph_dev, reverse_graph_dev, data_dev, data_power_dev, it,
                                              reverse_distance, reverse_num, new_num_global, old_num_global,
                                              hybrid_list, dim, k);
         nn_descent_opt_merge<<<grid, block3>>>(graph_dev, reverse_graph_dev, it, all_it, nei_distance,
                                                reverse_distance, nei_visit, reverse_num, k);
         reset_reverse_num<<<1000, 1024>>>(reverse_num, num_points);
      }
      do_reverse_graph<<<num_points, 32>>>(graph_dev, reverse_graph_dev, nei_distance, reverse_distance, reverse_num, k);
      reset_visit_reversenum<<<num_points, 32>>>(reverse_graph_dev, nei_visit, reverse_num, num_points, k);

      for (unsigned it = 0; it < all_it2; ++it)
      {
         sample_kernel6<<<grid, block>>>(graph_dev, reverse_graph_dev, it, data_dev, num_points, all_it2,
                                         nei_distance, reverse_distance, nei_visit, reverse_num, dim, k);
         if (it + 1 < all_it2)
            reset_reverse_num<<<1000, 1024>>>(reverse_num, num_points);
      }
      merge_reverse_plus<<<grid, block3>>>(graph_dev, reverse_graph_dev, nei_distance, reverse_distance, reverse_num, k);
      reset_reverse_num<<<1000, 1024>>>(reverse_num, num_points);
      ck(cudaDeviceSynchronize(), "GNN-Descent synchronize");
      result.gnn_ms = std::chrono::duration<double, std::milli>(
                          std::chrono::high_resolution_clock::now() - gnn_start)
                          .count();

      const auto prune_start = std::chrono::high_resolution_clock::now();
      funcFormat dis_func;
      ck(cudaMemcpyFromSymbol(&dis_func, tagore_dis_filter, sizeof(funcFormat)), "cudaMemcpyFromSymbol dis_filter");
      cal_ep_gpu<<<grid_one, block_s>>>(graph_dev, reverse_graph_dev, ep_dev, center_dev, data_dev, SELECT_CAND, dim,
                                        top_m, k);
      select_path<<<grid, block_s>>>(graph_dev, reverse_graph_dev, ep_dev, data_dev, SELECT_CAND, nei_distance,
                                     reverse_distance, reverse_num, dim, final_degree, top_m, k, alpha);
      filter_reverse<<<grid, block_s>>>(graph_dev, reverse_graph_dev, data_dev, nei_distance, reverse_distance,
                                        reverse_num, dim, final_degree, k, dis_func, alpha);
      ck(cudaDeviceSynchronize(), "Vamana pruning synchronize");
      result.prune_ms = elapsed_ms(prune_start);

      auto d2h_start = std::chrono::high_resolution_clock::now();
      ck(cudaMemcpy(result.graph.data(), graph_dev, static_cast<size_t>(num_points) * k * sizeof(unsigned),
                    cudaMemcpyDeviceToHost),
         "cudaMemcpy graph D2H");
      ck(cudaMemcpy(&result.entry_point, ep_dev, sizeof(unsigned), cudaMemcpyDeviceToHost), "cudaMemcpy ep D2H");
      result.d2h_ms = elapsed_ms(d2h_start);
   }
   catch (...)
   {
      cudaFree(data_float_dev);
      cudaFree(data_dev);
      cudaFree(center_dev);
      cudaFree(graph_dev);
      cudaFree(reverse_graph_dev);
      cudaFree(nei_distance);
      cudaFree(reverse_distance);
      cudaFree(nei_visit);
      cudaFree(reverse_num);
      cudaFree(reverse_num_old);
      cudaFree(new_num_global);
      cudaFree(old_num_global);
      cudaFree(hybrid_list);
      cudaFree(data_power_dev);
      cudaFree(ep_dev);
      throw;
   }

   auto free_start = std::chrono::high_resolution_clock::now();
   cudaFree(data_float_dev);
   cudaFree(data_dev);
   cudaFree(center_dev);
   cudaFree(graph_dev);
   cudaFree(reverse_graph_dev);
   cudaFree(nei_distance);
   cudaFree(reverse_distance);
   cudaFree(nei_visit);
   cudaFree(reverse_num);
   cudaFree(reverse_num_old);
   cudaFree(new_num_global);
   cudaFree(old_num_global);
   cudaFree(hybrid_list);
   cudaFree(data_power_dev);
   cudaFree(ep_dev);
   result.free_ms = elapsed_ms(free_start);

   return result;
}

static TagoreBatchBuildResult build_tagore_vamana_cuda_batch_impl(const std::vector<TagoreGroupRequest> &groups,
                                                                  uint32_t dim,
                                                                  uint32_t k,
                                                                  uint32_t final_degree,
                                                                  uint32_t top_m,
                                                                  uint32_t iterations,
                                                                  float alpha,
                                                                  TagorePruneMode prune_mode,
                                                                  const TagoreCudaRuntimeConfig &runtime_config)
{
   if (groups.empty())
      return {};
   if (dim == 0 || k == 0 || final_degree == 0 || top_m == 0)
      throw std::runtime_error("TagoreCuda batch requires positive dim/k/final_degree/top_m.");
   if (k <= final_degree)
      k = final_degree + 1;

   uint32_t max_points = 0;
   bool can_use_fast_exact_batch = prune_mode == TagorePruneMode::FastGrnnd;
   const uint32_t fast_exact_threshold = runtime_config.fast_exact_batch_threshold;
   for (const auto &g : groups)
   {
      if (!g.data || g.num_points == 0)
         throw std::runtime_error("TagoreCuda batch received an empty group.");
      max_points = std::max(max_points, g.num_points);
      if (fast_exact_threshold == 0 || g.num_points > fast_exact_threshold)
         can_use_fast_exact_batch = false;
   }

   if (can_use_fast_exact_batch)
      return build_fast_exact_cuda_batch(groups, dim, k, final_degree, runtime_config.fast_exact);

   const uint32_t requested_streams = runtime_config.requested_streams;
   if (requested_streams > 1 && groups.size() > 1)
   {
      const size_t workers = std::min<size_t>(requested_streams, groups.size());
      std::vector<std::vector<TagoreGroupRequest>> worker_groups(workers);
      std::vector<std::vector<size_t>> worker_positions(workers);
      std::vector<uint64_t> worker_points(workers, 0);
      std::vector<size_t> order(groups.size());
      for (size_t i = 0; i < groups.size(); ++i)
         order[i] = i;
      std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
         return groups[a].num_points > groups[b].num_points;
      });
      for (size_t pos : order)
      {
         const size_t target = static_cast<size_t>(
             std::min_element(worker_points.begin(), worker_points.end()) - worker_points.begin());
         worker_groups[target].push_back(groups[pos]);
         worker_positions[target].push_back(pos);
         worker_points[target] += groups[pos].num_points;
      }

      TagoreBatchBuildResult merged;
      merged.groups.resize(groups.size());
      std::vector<TagoreBatchBuildResult> partials(workers);
      std::vector<std::exception_ptr> errors(workers);
      std::vector<std::thread> threads;
      threads.reserve(workers);
      for (size_t wi = 0; wi < workers; ++wi)
      {
         threads.emplace_back([&, wi]() {
            try
            {
               partials[wi] = build_tagore_vamana_cuda_batch_impl(
                   worker_groups[wi],
                   dim,
                   k,
                   final_degree,
                   top_m,
                   iterations,
                   alpha,
                   prune_mode,
                   make_single_stream_config(runtime_config));
            }
            catch (...)
            {
               errors[wi] = std::current_exception();
            }
         });
      }
      for (auto &thread : threads)
         thread.join();
      for (const auto &err : errors)
      {
         if (err)
            std::rethrow_exception(err);
      }
      for (size_t wi = 0; wi < workers; ++wi)
      {
         merged.batch_pack_ms += partials[wi].batch_pack_ms;
         merged.batch_alloc_ms += partials[wi].batch_alloc_ms;
         merged.batch_free_ms += partials[wi].batch_free_ms;
         for (size_t i = 0; i < worker_positions[wi].size(); ++i)
            merged.groups[worker_positions[wi][i]] = std::move(partials[wi].groups[i]);
      }
      return merged;
   }

   TagoreBatchBuildResult batch;
   batch.groups.resize(groups.size());

   float *data_float_dev = nullptr;
   half *data_dev = nullptr;
   half *center_dev = nullptr;
   unsigned *graph_dev = nullptr;
   unsigned *reverse_graph_dev = nullptr;
   float *nei_distance = nullptr;
   float *reverse_distance = nullptr;
   bool *nei_visit = nullptr;
   unsigned *reverse_num = nullptr;
   unsigned *reverse_num_old = nullptr;
   unsigned *new_num_global = nullptr;
   unsigned *old_num_global = nullptr;
   unsigned *hybrid_list = nullptr;
   float *data_power_dev = nullptr;
   unsigned *ep_dev = nullptr;
   uint32_t *compact_graph_dev = nullptr;
   unsigned *fast_prune_reverse_dev = nullptr;
   unsigned *fast_prune_reverse_num_dev = nullptr;
   uint32_t fast_prune_reverse_cap = 0;
   cudaStream_t stream = nullptr;
   cudaEvent_t stage_h2d_begin = nullptr;
   cudaEvent_t stage_h2d_end = nullptr;
   cudaEvent_t stage_convert_end = nullptr;
   cudaEvent_t stage_memset_end = nullptr;
   cudaEvent_t stage_gnn_end = nullptr;
   cudaEvent_t stage_prune_end = nullptr;
   cudaEvent_t stage_d2h_end = nullptr;

   try
   {
      ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate Tagore batch");
      const uint32_t compact_stride = final_degree + 1;
      const bool compact_d2h = runtime_config.compact_d2h_requested && compact_stride < k;
      auto alloc_start = std::chrono::high_resolution_clock::now();
      ck(cudaMallocAsync(&data_float_dev, static_cast<size_t>(max_points) * dim * sizeof(float), stream),
         "cudaMallocAsync data_float_dev");
      ck(cudaMallocAsync(&data_dev, static_cast<size_t>(max_points) * dim * sizeof(half), stream), "cudaMallocAsync data_dev");
      ck(cudaMallocAsync(&center_dev, static_cast<size_t>(dim) * sizeof(half), stream), "cudaMallocAsync center_dev");
      ck(cudaMallocAsync(&graph_dev, static_cast<size_t>(max_points) * k * sizeof(unsigned), stream), "cudaMallocAsync graph_dev");
      ck(cudaMallocAsync(&reverse_graph_dev, static_cast<size_t>(max_points) * RESERVENUM * sizeof(unsigned), stream),
         "cudaMallocAsync reverse_graph_dev");
      ck(cudaMallocAsync(&nei_distance, static_cast<size_t>(max_points) * k * sizeof(float), stream), "cudaMallocAsync nei_distance");
      ck(cudaMallocAsync(&reverse_distance, static_cast<size_t>(max_points) * RESERVENUM * sizeof(float), stream),
         "cudaMallocAsync reverse_distance");
      ck(cudaMallocAsync(&nei_visit, static_cast<size_t>(max_points) * k * sizeof(bool), stream), "cudaMallocAsync nei_visit");
      ck(cudaMallocAsync(&reverse_num, static_cast<size_t>(max_points) * sizeof(unsigned), stream), "cudaMallocAsync reverse_num");
      ck(cudaMallocAsync(&reverse_num_old, static_cast<size_t>(max_points) * sizeof(unsigned), stream), "cudaMallocAsync reverse_num_old");
      ck(cudaMallocAsync(&new_num_global, static_cast<size_t>(max_points) * sizeof(unsigned), stream), "cudaMallocAsync new_num_global");
      ck(cudaMallocAsync(&old_num_global, static_cast<size_t>(max_points) * sizeof(unsigned), stream), "cudaMallocAsync old_num_global");
      ck(cudaMallocAsync(&hybrid_list, static_cast<size_t>(max_points) * SAMPLE * 4 * sizeof(unsigned), stream),
         "cudaMallocAsync hybrid_list");
      ck(cudaMallocAsync(&data_power_dev, static_cast<size_t>(max_points) * sizeof(float), stream), "cudaMallocAsync data_power_dev");
      ck(cudaMallocAsync(&ep_dev, sizeof(unsigned), stream), "cudaMallocAsync ep_dev");
      if (compact_d2h)
         ck(cudaMallocAsync(&compact_graph_dev, static_cast<size_t>(max_points) * compact_stride * sizeof(uint32_t), stream),
            "cudaMallocAsync compact_graph_dev");
      if (prune_mode == TagorePruneMode::FastGrnnd)
      {
         fast_prune_reverse_cap = std::max<uint32_t>(1, final_degree);
         ck(cudaMallocAsync(&fast_prune_reverse_dev,
                            static_cast<size_t>(max_points) * fast_prune_reverse_cap * sizeof(unsigned), stream),
            "cudaMallocAsync fast prune reverse workspace");
         ck(cudaMallocAsync(&fast_prune_reverse_num_dev, static_cast<size_t>(max_points) * sizeof(unsigned), stream),
            "cudaMallocAsync fast prune reverse count workspace");
      }
      ck(cudaStreamSynchronize(stream), "cudaMallocAsync synchronize");
      batch.batch_alloc_ms = elapsed_ms(alloc_start);

      ck(cudaEventCreate(&stage_h2d_begin), "cudaEventCreate h2d_begin");
      ck(cudaEventCreate(&stage_h2d_end), "cudaEventCreate h2d_end");
      ck(cudaEventCreate(&stage_convert_end), "cudaEventCreate convert_end");
      ck(cudaEventCreate(&stage_memset_end), "cudaEventCreate memset_end");
      ck(cudaEventCreate(&stage_gnn_end), "cudaEventCreate gnn_end");
      ck(cudaEventCreate(&stage_prune_end), "cudaEventCreate prune_end");
      ck(cudaEventCreate(&stage_d2h_end), "cudaEventCreate d2h_end");

      for (size_t gi = 0; gi < groups.size(); ++gi)
      {
         const float *data = groups[gi].data;
         const uint32_t num_points = groups[gi].num_points;
         auto &result = batch.groups[gi];
         const uint32_t result_stride = compact_d2h ? compact_stride : k;
         result.graph_stride = result_stride;
         result.graph.resize(static_cast<size_t>(num_points) * result_stride);
         result.alloc_ms = 0.0;
         result.free_ms = 0.0;

         const float norm_factor = compute_norm_factor(data, num_points, dim);

         ck(cudaEventRecord(stage_h2d_begin, stream), "cudaEventRecord h2d_begin");
         ck(cudaMemcpyAsync(data_float_dev, data, static_cast<size_t>(num_points) * dim * sizeof(float),
                            cudaMemcpyHostToDevice, stream),
            "cudaMemcpyAsync data_float_dev");
         ck(cudaEventRecord(stage_h2d_end, stream), "cudaEventRecord h2d_end");

         const size_t total_values = static_cast<size_t>(num_points) * dim;
         const int threads = 256;
         const int blocks = static_cast<int>(std::min<size_t>((total_values + threads - 1) / threads, 65535));
         convert_float_to_half_kernel<<<blocks, threads, 0, stream>>>(data_float_dev, data_dev, total_values, norm_factor);
         compute_center_half_kernel<<<dim, threads, 0, stream>>>(data_float_dev, center_dev, num_points, dim, norm_factor);
         ck(cudaGetLastError(), "float-to-half conversion launch");
         ck(cudaEventRecord(stage_convert_end, stream), "cudaEventRecord convert_end");

         ck(cudaMemsetAsync(reverse_num, 0, static_cast<size_t>(num_points) * sizeof(unsigned), stream), "memset reverse_num");
         ck(cudaMemsetAsync(reverse_num_old, 0, static_cast<size_t>(num_points) * sizeof(unsigned), stream), "memset reverse_num_old");
         ck(cudaMemsetAsync(new_num_global, 0, static_cast<size_t>(num_points) * sizeof(unsigned), stream), "memset new_num_global");
         ck(cudaMemsetAsync(old_num_global, 0, static_cast<size_t>(num_points) * sizeof(unsigned), stream), "memset old_num_global");
         ck(cudaMemsetAsync(data_power_dev, 0, static_cast<size_t>(num_points) * sizeof(float), stream), "memset data_power_dev");
         ck(cudaEventRecord(stage_memset_end, stream), "cudaEventRecord memset_end");

         dim3 grid(num_points, 1, 1);
         dim3 block(32, MAX_P / 32, 1);
         dim3 block2(32, 16, 1);
         dim3 block3(32, 3, 1);
         dim3 block_s(32, 4, 1);
         dim3 grid_one(1, 1, 1);

         initialize_graph<<<num_points, 32, 0, stream>>>(graph_dev, num_points, nei_distance, nei_visit, k);
         cal_power<<<num_points, dim, 0, stream>>>(data_dev, data_power_dev, dim, dim);

         const unsigned all_it = iterations / 2;
         const unsigned all_it2 = iterations / 2;
         for (unsigned it = 0; it < all_it; ++it)
         {
            nn_descent_opt_sample<<<num_points, 32, 0, stream>>>(graph_dev, reverse_graph_dev, nei_visit, reverse_num,
                                                                 reverse_num_old, new_num_global, old_num_global, hybrid_list, k);
            nn_descent_opt_reverse_sample<<<num_points, 32, 0, stream>>>(graph_dev, reverse_graph_dev, reverse_num, reverse_num_old,
                                                                         new_num_global, old_num_global, hybrid_list, k);
            reset_reverse_new_old_num<<<1000, 1024, 0, stream>>>(reverse_num, reverse_num_old, num_points);
            nn_descent_opt_cal<<<grid, block2, 0, stream>>>(graph_dev, reverse_graph_dev, data_dev, data_power_dev, it,
                                                            reverse_distance, reverse_num, new_num_global, old_num_global,
                                                            hybrid_list, dim, k);
            nn_descent_opt_merge<<<grid, block3, 0, stream>>>(graph_dev, reverse_graph_dev, it, all_it, nei_distance,
                                                              reverse_distance, nei_visit, reverse_num, k);
            reset_reverse_num<<<1000, 1024, 0, stream>>>(reverse_num, num_points);
         }
         do_reverse_graph<<<num_points, 32, 0, stream>>>(graph_dev, reverse_graph_dev, nei_distance, reverse_distance, reverse_num, k);
         reset_visit_reversenum<<<num_points, 32, 0, stream>>>(reverse_graph_dev, nei_visit, reverse_num, num_points, k);

         for (unsigned it = 0; it < all_it2; ++it)
         {
            sample_kernel6<<<grid, block, 0, stream>>>(graph_dev, reverse_graph_dev, it, data_dev, num_points, all_it2,
                                                       nei_distance, reverse_distance, nei_visit, reverse_num, dim, k);
            if (it + 1 < all_it2)
               reset_reverse_num<<<1000, 1024, 0, stream>>>(reverse_num, num_points);
         }
         merge_reverse_plus<<<grid, block3, 0, stream>>>(graph_dev, reverse_graph_dev, nei_distance, reverse_distance, reverse_num, k);
         reset_reverse_num<<<1000, 1024, 0, stream>>>(reverse_num, num_points);
         ck(cudaGetLastError(), "GNN-Descent launch");
         ck(cudaEventRecord(stage_gnn_end, stream), "cudaEventRecord gnn_end");

         if (prune_mode == TagorePruneMode::FastGrnnd)
         {
            launch_fast_grnnd_prune_impl(graph_dev, data_dev, nei_distance, num_points, dim, k, final_degree, alpha,
                                          stream, fast_prune_reverse_dev, fast_prune_reverse_num_dev,
                                          fast_prune_reverse_cap, false, runtime_config.fast_prune, result.prune_ms);
            ck(cudaMemsetAsync(ep_dev, 0, sizeof(unsigned), stream), "cudaMemset fast GRNND ep");
         }
         else
         {
            const auto prune_start = std::chrono::high_resolution_clock::now();
            funcFormat dis_func;
            ck(cudaMemcpyFromSymbol(&dis_func, tagore_dis_filter, sizeof(funcFormat)), "cudaMemcpyFromSymbol dis_filter");
            cal_ep_gpu<<<grid_one, block_s, 0, stream>>>(graph_dev, reverse_graph_dev, ep_dev, center_dev, data_dev, SELECT_CAND, dim,
                                                         top_m, k);
            select_path<<<grid, block_s, 0, stream>>>(graph_dev, reverse_graph_dev, ep_dev, data_dev, SELECT_CAND, nei_distance,
                                                      reverse_distance, reverse_num, dim, final_degree, top_m, k, alpha);
            filter_reverse<<<grid, block_s, 0, stream>>>(graph_dev, reverse_graph_dev, data_dev, nei_distance, reverse_distance,
                                                         reverse_num, dim, final_degree, k, dis_func, alpha);
            ck(cudaGetLastError(), "Vamana pruning launch");
            result.prune_ms = elapsed_ms(prune_start);
         }
         ck(cudaEventRecord(stage_prune_end, stream), "cudaEventRecord prune_end");

         if (prune_mode == TagorePruneMode::TagoreVamanaWithGrnndRefine)
            launch_grnnd_like_refine(graph_dev, data_dev, num_points, dim, k, final_degree, alpha,
                                     result.grnnd_refine_ms);

         const void *graph_copy_src = graph_dev;
         size_t graph_copy_bytes = static_cast<size_t>(num_points) * k * sizeof(unsigned);
         if (compact_d2h)
         {
            compact_graph_rows_kernel<<<num_points, 64, 0, stream>>>(graph_dev, compact_graph_dev,
                                                                     num_points, k, compact_stride, final_degree);
            ck(cudaGetLastError(), "compact graph rows launch");
            graph_copy_src = compact_graph_dev;
            graph_copy_bytes = static_cast<size_t>(num_points) * compact_stride * sizeof(uint32_t);
         }

         ck(cudaMemcpyAsync(result.graph.data(), graph_copy_src, graph_copy_bytes,
                            cudaMemcpyDeviceToHost, stream),
            "cudaMemcpyAsync graph D2H");
         ck(cudaMemcpyAsync(&result.entry_point, ep_dev, sizeof(unsigned), cudaMemcpyDeviceToHost, stream),
            "cudaMemcpyAsync ep D2H");
         ck(cudaEventRecord(stage_d2h_end, stream), "cudaEventRecord d2h_end");
         ck(cudaEventSynchronize(stage_d2h_end), "cudaEventSynchronize d2h_end");
         result.h2d_ms = cuda_event_elapsed_ms(stage_h2d_begin, stage_h2d_end, "cudaEventElapsed h2d");
         result.convert_ms = cuda_event_elapsed_ms(stage_h2d_end, stage_convert_end, "cudaEventElapsed convert");
         result.memset_ms = cuda_event_elapsed_ms(stage_convert_end, stage_memset_end, "cudaEventElapsed memset");
         result.gnn_ms = cuda_event_elapsed_ms(stage_memset_end, stage_gnn_end, "cudaEventElapsed gnn");
         result.prune_ms = cuda_event_elapsed_ms(stage_gnn_end, stage_prune_end, "cudaEventElapsed prune");
         result.d2h_ms = cuda_event_elapsed_ms(stage_prune_end, stage_d2h_end, "cudaEventElapsed d2h");
      }
   }
   catch (...)
   {
      cudaFreeAsync(data_float_dev, stream);
      cudaFreeAsync(data_dev, stream);
      cudaFreeAsync(center_dev, stream);
      cudaFreeAsync(graph_dev, stream);
      cudaFreeAsync(reverse_graph_dev, stream);
      cudaFreeAsync(nei_distance, stream);
      cudaFreeAsync(reverse_distance, stream);
      cudaFreeAsync(nei_visit, stream);
      cudaFreeAsync(reverse_num, stream);
      cudaFreeAsync(reverse_num_old, stream);
      cudaFreeAsync(new_num_global, stream);
      cudaFreeAsync(old_num_global, stream);
      cudaFreeAsync(hybrid_list, stream);
      cudaFreeAsync(data_power_dev, stream);
      cudaFreeAsync(ep_dev, stream);
      cudaFreeAsync(compact_graph_dev, stream);
      cudaFreeAsync(fast_prune_reverse_dev, stream);
      cudaFreeAsync(fast_prune_reverse_num_dev, stream);
      cudaStreamSynchronize(stream);
      if (stream)
         cudaStreamDestroy(stream);
      if (stage_h2d_begin)
         cudaEventDestroy(stage_h2d_begin);
      if (stage_h2d_end)
         cudaEventDestroy(stage_h2d_end);
      if (stage_convert_end)
         cudaEventDestroy(stage_convert_end);
      if (stage_memset_end)
         cudaEventDestroy(stage_memset_end);
      if (stage_gnn_end)
         cudaEventDestroy(stage_gnn_end);
      if (stage_prune_end)
         cudaEventDestroy(stage_prune_end);
      if (stage_d2h_end)
         cudaEventDestroy(stage_d2h_end);
      throw;
   }

   auto free_start = std::chrono::high_resolution_clock::now();
   ck(cudaFreeAsync(data_float_dev, stream), "cudaFreeAsync data_float_dev");
   ck(cudaFreeAsync(data_dev, stream), "cudaFreeAsync data_dev");
   ck(cudaFreeAsync(center_dev, stream), "cudaFreeAsync center_dev");
   ck(cudaFreeAsync(graph_dev, stream), "cudaFreeAsync graph_dev");
   ck(cudaFreeAsync(reverse_graph_dev, stream), "cudaFreeAsync reverse_graph_dev");
   ck(cudaFreeAsync(nei_distance, stream), "cudaFreeAsync nei_distance");
   ck(cudaFreeAsync(reverse_distance, stream), "cudaFreeAsync reverse_distance");
   ck(cudaFreeAsync(nei_visit, stream), "cudaFreeAsync nei_visit");
   ck(cudaFreeAsync(reverse_num, stream), "cudaFreeAsync reverse_num");
   ck(cudaFreeAsync(reverse_num_old, stream), "cudaFreeAsync reverse_num_old");
   ck(cudaFreeAsync(new_num_global, stream), "cudaFreeAsync new_num_global");
   ck(cudaFreeAsync(old_num_global, stream), "cudaFreeAsync old_num_global");
   ck(cudaFreeAsync(hybrid_list, stream), "cudaFreeAsync hybrid_list");
   ck(cudaFreeAsync(data_power_dev, stream), "cudaFreeAsync data_power_dev");
   ck(cudaFreeAsync(ep_dev, stream), "cudaFreeAsync ep_dev");
   if (compact_graph_dev)
      ck(cudaFreeAsync(compact_graph_dev, stream), "cudaFreeAsync compact_graph_dev");
   if (fast_prune_reverse_dev)
      ck(cudaFreeAsync(fast_prune_reverse_dev, stream), "cudaFreeAsync fast prune reverse workspace");
   if (fast_prune_reverse_num_dev)
      ck(cudaFreeAsync(fast_prune_reverse_num_dev, stream), "cudaFreeAsync fast prune reverse count workspace");
   ck(cudaStreamSynchronize(stream), "cudaFreeAsync synchronize");
   batch.batch_free_ms = elapsed_ms(free_start);

   cudaEventDestroy(stage_h2d_begin);
   cudaEventDestroy(stage_h2d_end);
   cudaEventDestroy(stage_convert_end);
   cudaEventDestroy(stage_memset_end);
   cudaEventDestroy(stage_gnn_end);
   cudaEventDestroy(stage_prune_end);
   cudaEventDestroy(stage_d2h_end);
   ck(cudaStreamDestroy(stream), "cudaStreamDestroy Tagore batch");

   return batch;
}

TagoreBatchBuildResult build_tagore_vamana_cuda_batch(const std::vector<TagoreGroupRequest> &groups,
                                                      uint32_t dim,
                                                      uint32_t k,
                                                      uint32_t final_degree,
                                                      uint32_t top_m,
                                                      uint32_t iterations,
                                                      float alpha,
                                                      TagorePruneMode prune_mode,
                                                      const TagoreCudaRuntimeConfig &runtime_config)
{
   return build_tagore_vamana_cuda_batch_impl(groups, dim, k, final_degree, top_m, iterations, alpha,
                                             prune_mode, runtime_config);
}

TagoreBatchBuildResult build_tagore_vamana_cuda_batch(const std::vector<TagoreGroupRequest> &groups,
                                                      uint32_t dim,
                                                      uint32_t k,
                                                      uint32_t final_degree,
                                                      uint32_t top_m,
                                                      uint32_t iterations,
                                                      float alpha,
                                                      TagorePruneMode prune_mode)
{
   return build_tagore_vamana_cuda_batch(
       groups,
       dim,
       k,
       final_degree,
       top_m,
       iterations,
       alpha,
       prune_mode,
       make_tagore_cuda_runtime_config(k, final_degree, true));
}

} // namespace ANNS
