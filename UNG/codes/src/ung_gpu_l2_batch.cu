#include "include/ung_gpu_l2_batch.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <mutex>

namespace ANNS
{
namespace
{
struct ResidentBaseCache
{
   const float *host_base = nullptr;
   IdxType num_points = 0;
   IdxType dim = 0;
   float *device_base = nullptr;
   IdxType *device_ids = nullptr;
   float *device_query = nullptr;
   float *device_queries = nullptr;
   IdxType *device_query_ids = nullptr;
   float *device_out = nullptr;
   size_t id_capacity = 0;
   size_t query_capacity = 0;
   size_t query_id_capacity = 0;
};

ResidentBaseCache g_cache;
std::mutex g_cache_mutex;

void reset_cache_locked()
{
   if (g_cache.device_base)
      cudaFree(g_cache.device_base);
   if (g_cache.device_ids)
      cudaFree(g_cache.device_ids);
   if (g_cache.device_query)
      cudaFree(g_cache.device_query);
   if (g_cache.device_queries)
      cudaFree(g_cache.device_queries);
   if (g_cache.device_query_ids)
      cudaFree(g_cache.device_query_ids);
   if (g_cache.device_out)
      cudaFree(g_cache.device_out);
   g_cache = ResidentBaseCache{};
}

bool ensure_base_locked(const float *base_vectors, IdxType num_points, IdxType dim)
{
   if (g_cache.host_base == base_vectors && g_cache.num_points == num_points &&
       g_cache.dim == dim && g_cache.device_base != nullptr)
      return true;

   reset_cache_locked();
   const size_t bytes = static_cast<size_t>(num_points) * static_cast<size_t>(dim) * sizeof(float);
   if (cudaMalloc(&g_cache.device_base, bytes) != cudaSuccess)
   {
      reset_cache_locked();
      return false;
   }
   if (cudaMemcpy(g_cache.device_base, base_vectors, bytes, cudaMemcpyHostToDevice) != cudaSuccess)
   {
      reset_cache_locked();
      return false;
   }
   g_cache.host_base = base_vectors;
   g_cache.num_points = num_points;
   g_cache.dim = dim;
   return true;
}

bool ensure_workspace_locked(size_t count, IdxType dim)
{
   if (g_cache.device_query == nullptr)
   {
      if (cudaMalloc(&g_cache.device_query, static_cast<size_t>(dim) * sizeof(float)) != cudaSuccess)
      {
         reset_cache_locked();
         return false;
      }
   }
   if (count <= g_cache.id_capacity && g_cache.device_ids != nullptr && g_cache.device_out != nullptr)
      return true;

   if (g_cache.device_ids)
      cudaFree(g_cache.device_ids);
   if (g_cache.device_out)
      cudaFree(g_cache.device_out);
   g_cache.device_ids = nullptr;
   g_cache.device_out = nullptr;
   g_cache.id_capacity = 0;

   if (cudaMalloc(&g_cache.device_ids, count * sizeof(IdxType)) != cudaSuccess)
   {
      reset_cache_locked();
      return false;
   }
   if (cudaMalloc(&g_cache.device_out, count * sizeof(float)) != cudaSuccess)
   {
      reset_cache_locked();
      return false;
   }
   g_cache.id_capacity = count;
   return true;
}

bool ensure_query_batch_workspace_locked(size_t query_count, size_t candidate_count, IdxType dim)
{
   if (query_count > g_cache.query_capacity)
   {
      if (g_cache.device_queries)
         cudaFree(g_cache.device_queries);
      g_cache.device_queries = nullptr;
      g_cache.query_capacity = 0;
      if (cudaMalloc(&g_cache.device_queries,
                     query_count * static_cast<size_t>(dim) * sizeof(float)) != cudaSuccess)
      {
         reset_cache_locked();
         return false;
      }
      g_cache.query_capacity = query_count;
   }

   if (candidate_count > g_cache.query_id_capacity)
   {
      if (g_cache.device_query_ids)
         cudaFree(g_cache.device_query_ids);
      g_cache.device_query_ids = nullptr;
      g_cache.query_id_capacity = 0;
      if (cudaMalloc(&g_cache.device_query_ids, candidate_count * sizeof(IdxType)) != cudaSuccess)
      {
         reset_cache_locked();
         return false;
      }
      g_cache.query_id_capacity = candidate_count;
   }
   return true;
}

__global__ void l2_batch_kernel(const float *__restrict__ base,
                                const float *__restrict__ query,
                                const IdxType *__restrict__ ids,
                                size_t count,
                                IdxType dim,
                                float *__restrict__ out)
{
   constexpr int WARP = 32;
   const int lane = threadIdx.x & (WARP - 1);
   const int warp_id = threadIdx.x / WARP;
   const int warps_per_block = blockDim.x / WARP;
   const size_t item = static_cast<size_t>(blockIdx.x) * warps_per_block + warp_id;
   if (item >= count)
      return;

   const IdxType id = ids[item];
   const float *vec = base + static_cast<size_t>(id) * dim;
   float acc = 0.0f;
   for (IdxType d = static_cast<IdxType>(lane); d < dim; d += WARP)
   {
      const float diff = query[d] - vec[d];
      acc += diff * diff;
   }
#pragma unroll
   for (int offset = 16; offset > 0; offset >>= 1)
      acc += __shfl_down_sync(0xffffffff, acc, offset);
   if (lane == 0)
      out[item] = acc;
}

__global__ void l2_query_candidate_batch_kernel(const float *__restrict__ base,
                                                const float *__restrict__ queries,
                                                const IdxType *__restrict__ ids,
                                                const IdxType *__restrict__ query_ids,
                                                size_t count,
                                                IdxType dim,
                                                float *__restrict__ out)
{
   constexpr int WARP = 32;
   const int lane = threadIdx.x & (WARP - 1);
   const int warp_id = threadIdx.x / WARP;
   const int warps_per_block = blockDim.x / WARP;
   const size_t item = static_cast<size_t>(blockIdx.x) * warps_per_block + warp_id;
   if (item >= count)
      return;

   const IdxType point_id = ids[item];
   const IdxType query_id = query_ids[item];
   const float *vec = base + static_cast<size_t>(point_id) * dim;
   const float *query = queries + static_cast<size_t>(query_id) * dim;
   float acc = 0.0f;
   for (IdxType d = static_cast<IdxType>(lane); d < dim; d += WARP)
   {
      const float diff = query[d] - vec[d];
      acc += diff * diff;
   }
#pragma unroll
   for (int offset = 16; offset > 0; offset >>= 1)
      acc += __shfl_down_sync(0xffffffff, acc, offset);
   if (lane == 0)
      out[item] = acc;
}

} // namespace

bool gpu_l2_batch_compute_float(const float *base_vectors,
                                IdxType num_points,
                                IdxType dim,
                                const float *query,
                                const IdxType *ids,
                                size_t count,
                                float *out_distances)
{
   if (base_vectors == nullptr || query == nullptr || ids == nullptr || out_distances == nullptr ||
       num_points == 0 || dim == 0 || count == 0)
      return false;

   std::lock_guard<std::mutex> lock(g_cache_mutex);
   if (!ensure_base_locked(base_vectors, num_points, dim))
      return false;
   if (!ensure_workspace_locked(count, dim))
      return false;

   if (cudaMemcpy(g_cache.device_query, query, static_cast<size_t>(dim) * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
      return false;
   if (cudaMemcpy(g_cache.device_ids, ids, count * sizeof(IdxType), cudaMemcpyHostToDevice) != cudaSuccess)
      return false;

   constexpr int threads = 128;
   constexpr int warps_per_block = threads / 32;
   const int blocks = static_cast<int>((count + warps_per_block - 1) / warps_per_block);
   l2_batch_kernel<<<blocks, threads>>>(g_cache.device_base, g_cache.device_query,
                                        g_cache.device_ids, count, dim, g_cache.device_out);
   if (cudaGetLastError() != cudaSuccess)
      return false;
   if (cudaMemcpy(out_distances, g_cache.device_out, count * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
      return false;
   return cudaDeviceSynchronize() == cudaSuccess;
}


bool gpu_l2_batch_compute_query_candidates_float(const float *base_vectors,
                                                 IdxType num_points,
                                                 IdxType dim,
                                                 const float *query_vectors,
                                                 size_t query_count,
                                                 const IdxType *candidate_ids,
                                                 const IdxType *candidate_query_ids,
                                                 size_t candidate_count,
                                                 float *out_distances)
{
   if (base_vectors == nullptr || query_vectors == nullptr || candidate_ids == nullptr ||
       candidate_query_ids == nullptr || out_distances == nullptr || num_points == 0 ||
       dim == 0 || query_count == 0 || candidate_count == 0)
      return false;

   std::lock_guard<std::mutex> lock(g_cache_mutex);
   if (!ensure_base_locked(base_vectors, num_points, dim))
      return false;
   if (!ensure_workspace_locked(candidate_count, dim))
      return false;

   const size_t query_bytes = query_count * static_cast<size_t>(dim) * sizeof(float);
   if (!ensure_query_batch_workspace_locked(query_count, candidate_count, dim))
      return false;

   bool ok = true;
   ok = ok && cudaMemcpy(g_cache.device_queries, query_vectors, query_bytes, cudaMemcpyHostToDevice) == cudaSuccess;
   ok = ok && cudaMemcpy(g_cache.device_ids, candidate_ids, candidate_count * sizeof(IdxType), cudaMemcpyHostToDevice) == cudaSuccess;
   ok = ok && cudaMemcpy(g_cache.device_query_ids, candidate_query_ids, candidate_count * sizeof(IdxType), cudaMemcpyHostToDevice) == cudaSuccess;
   if (ok)
   {
      constexpr int threads = 128;
      constexpr int warps_per_block = threads / 32;
      const int blocks = static_cast<int>((candidate_count + warps_per_block - 1) / warps_per_block);
      l2_query_candidate_batch_kernel<<<blocks, threads>>>(g_cache.device_base, g_cache.device_queries,
                                                           g_cache.device_ids, g_cache.device_query_ids,
                                                           candidate_count, dim, g_cache.device_out);
      ok = cudaGetLastError() == cudaSuccess;
   }
   if (ok)
      ok = cudaMemcpy(out_distances, g_cache.device_out, candidate_count * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess;
   if (ok)
      ok = cudaDeviceSynchronize() == cudaSuccess;

   return ok;
}

} // namespace ANNS
