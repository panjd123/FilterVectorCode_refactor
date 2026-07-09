#include "include/ung_gpu_cover_frontier_provider.h"

#include "include/ung_query_route.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace ANNS
{
namespace
{

using DeviceId = IdxType;
using DeviceLabel = LabelType;

constexpr DeviceLabel kInvalidDenseLabel = std::numeric_limits<DeviceLabel>::max();
constexpr DeviceId kDefaultFrontierDelta = 2;
constexpr DeviceId kDefaultFrontierCoverCap = 8192;

void cuda_check(cudaError_t status, const char *expr, const char *file, int line)
{
   if (status == cudaSuccess)
      return;
   throw std::runtime_error(std::string("CUDA error: ") + expr + " failed at " +
                            file + ":" + std::to_string(line) + ": " +
                            cudaGetErrorString(status));
}

#define ANNS_CUDA_CHECK(expr) cuda_check((expr), #expr, __FILE__, __LINE__)

__global__ void gpu_cover_candidate_frontier_kernel(const uint64_t *label_group_bits,
                                                    const uint64_t *size_group_bits,
                                                    DeviceId words_per_query,
                                                    DeviceId max_group_label_size,
                                                    const DeviceLabel *query_dense_labels,
                                                    const DeviceId *query_offsets,
                                                    DeviceId num_queries_chunk,
                                                    DeviceId frontier_delta,
                                                    uint64_t *candidate_bits,
                                                    uint64_t *frontier_bits)
{
   DeviceId word = blockIdx.x * blockDim.x + threadIdx.x;
   DeviceId local_q = blockIdx.y;
   if (local_q >= num_queries_chunk || word >= words_per_query)
      return;

   DeviceId q_begin = query_offsets[local_q];
   DeviceId q_end = query_offsets[local_q + 1];
   uint64_t candidates = ~uint64_t{0};
   for (DeviceId off = q_begin; off < q_end; ++off)
   {
      DeviceLabel dense = query_dense_labels[off];
      if (dense == kInvalidDenseLabel)
      {
         candidates = 0;
         break;
      }
      candidates &= label_group_bits[static_cast<size_t>(dense) * words_per_query + word];
   }

   DeviceId q_len = q_end - q_begin;
   DeviceId end_len = min(max_group_label_size, q_len + frontier_delta);
   uint64_t size_mask = 0;
   for (DeviceId len = q_len; len <= end_len; ++len)
      size_mask |= size_group_bits[static_cast<size_t>(len) * words_per_query + word];

   const size_t out = static_cast<size_t>(local_q) * words_per_query + word;
   candidate_bits[out] = candidates;
   frontier_bits[out] = candidates & size_mask;
}

__global__ void gpu_cover_compact_frontier_ids_kernel(const uint64_t *frontier_bits,
                                                      DeviceId num_groups_including_zero,
                                                      DeviceId words_per_query,
                                                      DeviceId num_queries_chunk,
                                                      DeviceId frontier_cover_cap,
                                                      uint64_t *selected_frontier_bits,
                                                      DeviceId *frontier_counts,
                                                      DeviceId *frontier_ids)
{
   DeviceId local_q = blockIdx.x;
   if (local_q >= num_queries_chunk)
      return;

   const uint64_t *frontier = frontier_bits + static_cast<size_t>(local_q) * words_per_query;
   __shared__ DeviceId count;
   if (threadIdx.x == 0)
      count = 0;
   __syncthreads();

   for (DeviceId fword = threadIdx.x; fword < words_per_query; fword += blockDim.x)
   {
      uint64_t bits = frontier[fword];
      while (bits)
      {
         DeviceId bit = static_cast<DeviceId>(__ffsll(static_cast<long long>(bits)) - 1);
         DeviceId gid = (fword << 6) + bit;
         if (gid > 0 && gid < num_groups_including_zero)
         {
            DeviceId slot = count;
            if (slot < frontier_cover_cap)
               slot = atomicAdd(&count, 1u);
            if (slot < frontier_cover_cap)
            {
               frontier_ids[static_cast<size_t>(local_q) * frontier_cover_cap + slot] = gid;
               atomicOr(reinterpret_cast<unsigned long long *>(
                            selected_frontier_bits + static_cast<size_t>(local_q) * words_per_query + fword),
                        1ull << bit);
            }
         }
         bits &= bits - 1;
      }
   }

   __syncthreads();
   if (threadIdx.x == 0)
      frontier_counts[local_q] = min(count, frontier_cover_cap);
}

__global__ void gpu_cover_descendant_cover_list_kernel(const DeviceId *frontier_counts,
                                                       const DeviceId *frontier_ids,
                                                       const uint64_t *group_desc_bits,
                                                       DeviceId words_per_query,
                                                       DeviceId num_queries_chunk,
                                                       DeviceId frontier_cover_cap,
                                                       uint64_t *covered_bits)
{
   DeviceId word = blockIdx.x * blockDim.x + threadIdx.x;
   DeviceId local_q = blockIdx.y;
   if (local_q >= num_queries_chunk || word >= words_per_query)
      return;

   DeviceId count = frontier_counts[local_q];
   const DeviceId *ids = frontier_ids + static_cast<size_t>(local_q) * frontier_cover_cap;
   uint64_t covered = 0;
   for (DeviceId i = 0; i < count; ++i)
   {
      DeviceId gid = ids[i];
      covered |= group_desc_bits[static_cast<size_t>(gid) * words_per_query + word];
   }
   covered_bits[static_cast<size_t>(local_q) * words_per_query + word] = covered;
}

__global__ void gpu_cover_select_kernel(const uint64_t *candidate_bits,
                                        const uint64_t *selected_frontier_bits,
                                        const uint64_t *covered_bits,
                                        DeviceId words_per_query,
                                        DeviceId num_queries_chunk,
                                        uint64_t *out_bits)
{
   DeviceId word = blockIdx.x * blockDim.x + threadIdx.x;
   DeviceId local_q = blockIdx.y;
   if (local_q >= num_queries_chunk || word >= words_per_query)
      return;

   const size_t idx = static_cast<size_t>(local_q) * words_per_query + word;
   out_bits[idx] = selected_frontier_bits[idx] | (candidate_bits[idx] & ~covered_bits[idx]);
}

void set_group_bit(std::vector<uint64_t> &bits, DeviceId words, DeviceId row, DeviceId group_id)
{
   bits[static_cast<size_t>(row) * words + (group_id >> 6)] |= uint64_t{1} << (group_id & 63);
}

} // namespace

struct GpuCoverFrontierProvider::Impl
{
   explicit Impl(const GpuCoverFrontierBuildInput &input)
   {
      if (input.group_labels == nullptr)
         throw std::invalid_argument("GpuCoverFrontierProvider requires group labels");
      if (input.num_groups == 0)
         throw std::invalid_argument("GpuCoverFrontierProvider requires non-empty groups");
      if (input.group_labels->size() <= input.num_groups)
         throw std::invalid_argument("GpuCoverFrontierProvider group label table is smaller than num_groups");

      num_groups_including_zero = input.num_groups + 1;
      words_per_query = (num_groups_including_zero + 63) / 64;

      for (DeviceId gid = 1; gid <= input.num_groups; ++gid)
      {
         const auto &labels = (*input.group_labels)[gid];
         max_group_label_size = std::max<DeviceId>(max_group_label_size, labels.size());
         for (LabelType label : labels)
         {
            if (label_to_dense.find(label) == label_to_dense.end())
            {
               const DeviceId dense = static_cast<DeviceId>(label_to_dense.size());
               label_to_dense.emplace(label, dense);
            }
         }
      }

      std::vector<uint64_t> label_group_bits(static_cast<size_t>(label_to_dense.size()) * words_per_query, 0);
      std::vector<uint64_t> size_group_bits(static_cast<size_t>(max_group_label_size + 1) * words_per_query, 0);
      for (DeviceId gid = 1; gid <= input.num_groups; ++gid)
      {
         const auto &labels = (*input.group_labels)[gid];
         set_group_bit(size_group_bits, words_per_query, static_cast<DeviceId>(labels.size()), gid);
         for (LabelType label : labels)
            set_group_bit(label_group_bits, words_per_query, label_to_dense.at(label), gid);
      }

      std::vector<uint64_t> descendant_bits(static_cast<size_t>(num_groups_including_zero) * words_per_query, 0);
      if (input.lng_descendants != nullptr && input.lng_descendants->size() > input.num_groups)
      {
         for (DeviceId gid = 1; gid <= input.num_groups; ++gid)
         {
            for (IdxType desc : (*input.lng_descendants)[gid])
               if (desc > 0 && desc <= input.num_groups)
                  set_group_bit(descendant_bits, words_per_query, gid, desc);
         }
      }
      else
      {
         for (DeviceId parent = 1; parent <= input.num_groups; ++parent)
         {
            const auto &parent_labels = (*input.group_labels)[parent];
            for (DeviceId child = 1; child <= input.num_groups; ++child)
            {
               if (child == parent)
                  continue;
               const auto &child_labels = (*input.group_labels)[child];
               if (child_labels.size() > parent_labels.size() &&
                   std::includes(child_labels.begin(), child_labels.end(),
                                 parent_labels.begin(), parent_labels.end()))
                  set_group_bit(descendant_bits, words_per_query, parent, child);
            }
         }
      }

      ANNS_CUDA_CHECK(cudaMalloc(&d_label_group_bits, label_group_bits.size() * sizeof(uint64_t)));
      ANNS_CUDA_CHECK(cudaMalloc(&d_size_group_bits, size_group_bits.size() * sizeof(uint64_t)));
      ANNS_CUDA_CHECK(cudaMalloc(&d_descendant_bits, descendant_bits.size() * sizeof(uint64_t)));
      ANNS_CUDA_CHECK(cudaMemcpy(d_label_group_bits, label_group_bits.data(),
                                 label_group_bits.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
      ANNS_CUDA_CHECK(cudaMemcpy(d_size_group_bits, size_group_bits.data(),
                                 size_group_bits.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
      ANNS_CUDA_CHECK(cudaMemcpy(d_descendant_bits, descendant_bits.data(),
                                 descendant_bits.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
   }

   ~Impl()
   {
      cudaFree(d_label_group_bits);
      cudaFree(d_size_group_bits);
      cudaFree(d_descendant_bits);
   }

   EntryGroupProviderResult run(const EntryGroupProviderRequest &request, QueryStats &stats) const
   {
      const auto start = std::chrono::high_resolution_clock::now();
      std::vector<DeviceLabel> dense_query_labels;
      dense_query_labels.reserve(request.query_labels->size());
      for (LabelType label : *request.query_labels)
      {
         auto it = label_to_dense.find(label);
         dense_query_labels.push_back(it == label_to_dense.end() ? kInvalidDenseLabel : it->second);
      }
      if (dense_query_labels.empty())
         dense_query_labels.push_back(kInvalidDenseLabel);

      const std::vector<DeviceId> query_offsets = {0, static_cast<DeviceId>(request.query_labels->size())};
      DeviceId *d_query_offsets = nullptr;
      DeviceLabel *d_query_labels = nullptr;
      uint64_t *d_candidate_bits = nullptr;
      uint64_t *d_frontier_bits = nullptr;
      uint64_t *d_selected_frontier_bits = nullptr;
      uint64_t *d_covered_bits = nullptr;
      uint64_t *d_out_bits = nullptr;
      DeviceId *d_frontier_counts = nullptr;
      DeviceId *d_frontier_ids = nullptr;

      const size_t words_bytes = static_cast<size_t>(words_per_query) * sizeof(uint64_t);
      ANNS_CUDA_CHECK(cudaMalloc(&d_query_offsets, query_offsets.size() * sizeof(DeviceId)));
      ANNS_CUDA_CHECK(cudaMalloc(&d_query_labels, dense_query_labels.size() * sizeof(DeviceLabel)));
      ANNS_CUDA_CHECK(cudaMalloc(&d_candidate_bits, words_bytes));
      ANNS_CUDA_CHECK(cudaMalloc(&d_frontier_bits, words_bytes));
      ANNS_CUDA_CHECK(cudaMalloc(&d_selected_frontier_bits, words_bytes));
      ANNS_CUDA_CHECK(cudaMalloc(&d_covered_bits, words_bytes));
      ANNS_CUDA_CHECK(cudaMalloc(&d_out_bits, words_bytes));
      ANNS_CUDA_CHECK(cudaMalloc(&d_frontier_counts, sizeof(DeviceId)));
      ANNS_CUDA_CHECK(cudaMalloc(&d_frontier_ids, static_cast<size_t>(kDefaultFrontierCoverCap) * sizeof(DeviceId)));

      ANNS_CUDA_CHECK(cudaMemcpy(d_query_offsets, query_offsets.data(),
                                 query_offsets.size() * sizeof(DeviceId), cudaMemcpyHostToDevice));
      ANNS_CUDA_CHECK(cudaMemcpy(d_query_labels, dense_query_labels.data(),
                                 dense_query_labels.size() * sizeof(DeviceLabel), cudaMemcpyHostToDevice));
      ANNS_CUDA_CHECK(cudaMemset(d_selected_frontier_bits, 0, words_bytes));

      dim3 block(256);
      dim3 grid((words_per_query + block.x - 1) / block.x, 1);
      gpu_cover_candidate_frontier_kernel<<<grid, block>>>(d_label_group_bits, d_size_group_bits,
                                                           words_per_query, max_group_label_size,
                                                           d_query_labels, d_query_offsets, 1,
                                                           kDefaultFrontierDelta,
                                                           d_candidate_bits, d_frontier_bits);
      ANNS_CUDA_CHECK(cudaGetLastError());
      gpu_cover_compact_frontier_ids_kernel<<<1, 256>>>(d_frontier_bits, num_groups_including_zero,
                                                        words_per_query, 1, kDefaultFrontierCoverCap,
                                                        d_selected_frontier_bits,
                                                        d_frontier_counts, d_frontier_ids);
      ANNS_CUDA_CHECK(cudaGetLastError());
      gpu_cover_descendant_cover_list_kernel<<<grid, block>>>(d_frontier_counts, d_frontier_ids,
                                                              d_descendant_bits, words_per_query,
                                                              1, kDefaultFrontierCoverCap,
                                                              d_covered_bits);
      ANNS_CUDA_CHECK(cudaGetLastError());
      gpu_cover_select_kernel<<<grid, block>>>(d_candidate_bits, d_selected_frontier_bits,
                                               d_covered_bits, words_per_query, 1, d_out_bits);
      ANNS_CUDA_CHECK(cudaGetLastError());

      std::vector<uint64_t> host_bits(words_per_query);
      ANNS_CUDA_CHECK(cudaMemcpy(host_bits.data(), d_out_bits, words_bytes, cudaMemcpyDeviceToHost));

      cudaFree(d_query_offsets);
      cudaFree(d_query_labels);
      cudaFree(d_candidate_bits);
      cudaFree(d_frontier_bits);
      cudaFree(d_selected_frontier_bits);
      cudaFree(d_covered_bits);
      cudaFree(d_out_bits);
      cudaFree(d_frontier_counts);
      cudaFree(d_frontier_ids);

      EntryGroupProviderResult result;
      result.requested_impl = request.impl;
      result.provider = EntryGroupProviderKind::GpuCoverFrontier;
      result.coverage_correct = true;
      result.exact_minimal = false;
      for (DeviceId word = 0; word < words_per_query; ++word)
      {
         uint64_t bits = host_bits[word];
         while (bits)
         {
            DeviceId bit = static_cast<DeviceId>(__builtin_ctzll(bits));
            DeviceId gid = (word << 6) + bit;
            if (gid > 0 && gid < num_groups_including_zero)
               result.group_ids.push_back(gid);
            bits &= bits - 1;
         }
      }
      result.elapsed_ms = std::chrono::duration<double, std::milli>(
                              std::chrono::high_resolution_clock::now() - start)
                              .count();
      stats.get_min_super_sets_time_ms = result.elapsed_ms;
      return result;
   }

   DeviceId num_groups_including_zero = 0;
   DeviceId words_per_query = 0;
   DeviceId max_group_label_size = 0;
   std::unordered_map<LabelType, DeviceId> label_to_dense;
   uint64_t *d_label_group_bits = nullptr;
   uint64_t *d_size_group_bits = nullptr;
   uint64_t *d_descendant_bits = nullptr;
};

GpuCoverFrontierProvider::GpuCoverFrontierProvider(const GpuCoverFrontierBuildInput &input)
    : impl_(std::make_unique<Impl>(input))
{
}

GpuCoverFrontierProvider::~GpuCoverFrontierProvider() = default;

EntryGroupProviderResult GpuCoverFrontierProvider::run(const EntryGroupProviderRequest &request,
                                                       QueryStats &stats) const
{
   return impl_->run(request, stats);
}

} // namespace ANNS
