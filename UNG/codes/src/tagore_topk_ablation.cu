#include "tagore_topk_ablation.cuh"
#include <Tagore_src.cuh>

namespace ANNS
{
namespace
{
constexpr unsigned kVisited = 0x80000000u;

__device__ bool merge_less(unsigned a_id, float a_dist, unsigned a_tag,
                           unsigned b_id, float b_dist, unsigned b_tag, bool by_id)
{
   if (by_id)
      return a_id < b_id || (a_id == b_id && (a_tag & ~kVisited) < (b_tag & ~kVisited));
   return a_dist < b_dist || (a_dist == b_dist &&
          (a_id < b_id || (a_id == b_id && (a_tag & ~kVisited) < (b_tag & ~kVisited))));
}

// Sort the entire old+new union, rather than filtering by the old Kth distance
// and binary-searching only competitive new candidates into the old list.
// Pad to a power of two, deduplicate new IDs (old entries take precedence),
// sort all entries by distance, then truncate. Old visit flags travel with IDs.
template <unsigned Capacity>
__device__ void full_sort_merge(unsigned *graph, float *distances, bool *visited,
                                const unsigned *new_ids, const float *new_distances,
                                unsigned new_count, unsigned src, unsigned k)
{
   __shared__ unsigned ids[Capacity], tags[Capacity];
   __shared__ float dist[Capacity];
   const unsigned tid = threadIdx.x + threadIdx.y * blockDim.x;
   const unsigned threads = blockDim.x * blockDim.y;
   unsigned length = 1;
   while (length < k + new_count)
      length <<= 1;
   for (unsigned i = tid; i < length; i += threads)
   {
      tags[i] = i;
      if (i < k)
      {
         ids[i] = graph[src * k + i];
         dist[i] = distances[src * k + i];
         if (visited && visited[src * k + i])
            tags[i] |= kVisited;
      }
      else if (i < k + new_count)
      {
         ids[i] = new_ids[i - k];
         dist[i] = ids[i] == src ? INF_DIS : new_distances[i - k];
      }
      else
      {
         ids[i] = 0xffffffffu;
         dist[i] = INF_DIS;
      }
   }
   __syncthreads();
   for (unsigned pass = 0; pass < 2; ++pass)
   {
      const bool by_id = pass == 0;
      for (unsigned width = 2; width <= length; width <<= 1)
      {
         for (unsigned step = width >> 1; step > 0; step >>= 1)
         {
            for (unsigned i = tid; i < length; i += threads)
            {
               const unsigned j = i ^ step;
               if (i < j)
               {
                  const bool exchange = (i & width) == 0
                      ? merge_less(ids[j], dist[j], tags[j], ids[i], dist[i], tags[i], by_id)
                      : merge_less(ids[i], dist[i], tags[i], ids[j], dist[j], tags[j], by_id);
                  if (exchange)
                  {
                     const unsigned id = ids[i], tag = tags[i];
                     const float d = dist[i];
                     ids[i] = ids[j]; dist[i] = dist[j]; tags[i] = tags[j];
                     ids[j] = id; dist[j] = d; tags[j] = tag;
                  }
               }
            }
            __syncthreads();
         }
      }
      if (by_id)
      {
         for (unsigned i = tid; i < length; i += threads)
            if ((tags[i] & ~kVisited) >= k && i > 0 && ids[i] == ids[i - 1])
               dist[i] = INF_DIS;
         __syncthreads();
      }
   }
   for (unsigned i = tid; i < k; i += threads)
   {
      graph[src * k + i] = ids[i];
      distances[src * k + i] = dist[i];
      if (visited)
         visited[src * k + i] = (tags[i] & kVisited) != 0;
   }
   __syncthreads();
}

__device__ unsigned ablation_uniform_random(unsigned long long seed, unsigned node_num)
{
   curandStatePhilox4_32_10_t state;
   const int idx = threadIdx.x + threadIdx.y * blockDim.x + blockIdx.x * blockDim.x * blockDim.y;
   curand_init(seed, idx, 1, &state);
   return static_cast<unsigned>(static_cast<float>(node_num) * curand_uniform(&state)) % node_num;
}
} // namespace

__global__ void tagore_nn_descent_full_sort(unsigned *graph, unsigned *reverse_graph,
                                           unsigned, unsigned, float *distances,
                                           float *reverse_distances, bool *visited,
                                           unsigned *reverse_count, unsigned k)
{
   const unsigned src = blockIdx.x;
   full_sort_merge<256>(graph, distances, visited,
                        reverse_graph + src * RESERVENUM,
                        reverse_distances + src * RESERVENUM,
                        min(static_cast<unsigned>(RESERVENUM), reverse_count[src]), src, k);
}

__global__ void tagore_reverse_full_sort(unsigned *graph, unsigned *reverse_graph,
                                        float *distances, float *reverse_distances,
                                        unsigned *reverse_count, unsigned k)
{
   const unsigned src = blockIdx.x;
   full_sort_merge<256>(graph, distances, nullptr,
                        reverse_graph + src * k, reverse_distances + src * RESERVENUM,
                        min(k, reverse_count[src]), src, k);
}

// Phase-two candidate generation below mirrors Tagore's sample_kernel6 through
// distance evaluation. Only the candidate-update/selection step is replaced.
__global__ void tagore_sample_full_sort(unsigned* graph, unsigned* reverse_graph, unsigned it, const half* __restrict__ values, unsigned node_num, unsigned all_it, float* nei_distance, float* reverse_distance, bool* nei_visit, unsigned* reverse_num, unsigned DIM, unsigned K){
    unsigned tmp_it = threadIdx.y/4;
    unsigned bid = blockIdx.x, laneid = threadIdx.x, tid = threadIdx.x + threadIdx.y * blockDim.x;
    __shared__ unsigned fetch_ids[4];
    __shared__ int lock;
    __shared__ unsigned new_list_shared[MAX_P], new_list2[MAX_P];
    __shared__ float new_dist_shared[MAX_P];
    __shared__ half4 tmp_val_sha[DIM_SIZE / 4];
    if(tid == 0) lock = 1;
    __syncthreads();
    if(threadIdx.y % 4 == 0 && laneid == 0){
        while (atomicExch(&lock, 0) == 0);
        while(nei_visit[bid*K+tmp_it] == true && tmp_it < K){
            tmp_it++;
        }
        nei_visit[bid*K+tmp_it] = true;
        lock = 1;
        fetch_ids[threadIdx.y / 4] = tmp_it % K;
    }
    __syncthreads();
    tmp_it = fetch_ids[threadIdx.y / 4];

    unsigned samp_id = graph[bid * K + tmp_it];
    if(threadIdx.y % 4 < 2){
        new_list_shared[tid] = graph[samp_id * K + laneid + (threadIdx.y%4)*32];
    }
    else{
        new_list_shared[tid] = reverse_graph[samp_id * K + ablation_uniform_random(clock(), K)];
    }


    for(unsigned i = tid; i < K; i += blockDim.x * blockDim.y){
        new_list2[i] = graph[bid * K + i];
    }
    __syncthreads();
    bitonic_sort_id_new2(new_list_shared, MAX_P);
    bitonic_sort_id_new2(new_list2, K);

    for(unsigned i = tid + 1; i < MAX_P; i += blockDim.x * blockDim.y){
        new_dist_shared[i] = (new_list_shared[i] == new_list_shared[i - 1] ? INF_DIS : (new_list_shared[i] == bid ? INF_DIS : 0));
    }
    if(tid == 0){
        if(new_list_shared[0] != bid){
            new_dist_shared[0] = 0;
        }
        else{
            new_dist_shared[0] = INF_DIS;
        }
    }
    __syncthreads();
    for(unsigned i = tid; i < MAX_P; i += blockDim.x * blockDim.y){
        if(new_dist_shared[i] > 1.0f) continue;
        unsigned tmp = K, res_id = 0;
        unsigned val = new_list_shared[i];
        while (tmp > 1) {
            unsigned halfsize = tmp / 2;
            unsigned cand = new_list2[res_id + halfsize];
            res_id += ((cand < val) ? halfsize : 0);
            tmp -= halfsize;
        }
        res_id += (new_list2[res_id] < val);
        if(new_list2[res_id] == new_list_shared[i]){
            new_dist_shared[i] = INF_DIS;
        }
    }
    __syncthreads();

    for(unsigned i = tid; i < (DIM / 4); i += blockDim.x * blockDim.y){
		tmp_val_sha[i] = Load(&values[bid * DIM + 4 * i]);
	}
	__syncthreads();
    #pragma unroll
    for(unsigned i = threadIdx.y; i < MAX_P; i += blockDim.y){
        if(new_dist_shared[i] < 1.0){
            half2 val_res;
            val_res.x = 0.0; val_res.y = 0.0;


            for(unsigned j = laneid; j < (DIM / 4); j += blockDim.x){
                half4 val2 = Load(&values[new_list_shared[i] * DIM + j * 4]);
                val_res = __hfma2(__hsub2(tmp_val_sha[j].x, val2.x), __hsub2(tmp_val_sha[j].x, val2.x), val_res);
                val_res = __hfma2(__hsub2(tmp_val_sha[j].y, val2.y), __hsub2(tmp_val_sha[j].y, val2.y), val_res);
            }
            #pragma unroll
            for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){
                val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
            }

            if(laneid == 0){
                new_dist_shared[i] = __half2float(__hadd(val_res.x, val_res.y));
            }
        }
    }
    __syncthreads();
    full_sort_merge<1024>(graph, nei_distance, nei_visit, new_list_shared,
                           new_dist_shared, MAX_P, bid, K);
    // Match the optimized kernel's reverse-edge propagation after selection.
    for (unsigned i = tid; i < K; i += blockDim.x * blockDim.y) {
        const unsigned id = graph[bid * K + i];
        const float distance = nei_distance[bid * K + i];
        const unsigned pos = atomicAdd(&reverse_num[id], 1);
        if (pos < K) {
            reverse_graph[id * K + pos] = bid;
            reverse_distance[id * RESERVENUM + pos] = distance;
        } else if (distance < reverse_distance[id * RESERVENUM + pos % K]) {
            reverse_graph[id * K + pos % K] = bid;
            reverse_distance[id * RESERVENUM + pos % K] = distance;
        }
    }
}
} // namespace ANNS
