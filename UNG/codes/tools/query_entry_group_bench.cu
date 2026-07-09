#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using Id = uint32_t;
using Label = uint32_t;
static constexpr Id kInvalidDenseLabel = std::numeric_limits<Id>::max();

#define CUDA_CHECK(expr)                                                                          \
   do                                                                                             \
   {                                                                                              \
      cudaError_t err__ = (expr);                                                                 \
      if (err__ != cudaSuccess)                                                                   \
      {                                                                                           \
         std::ostringstream oss__;                                                                \
         oss__ << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "                       \
               << cudaGetErrorString(err__);                                                      \
         throw std::runtime_error(oss__.str());                                                   \
      }                                                                                           \
   } while (0)

struct Options
{
   std::string base_label_file;
   std::string query_label_file;
   std::string provider = "all";
   std::string output_csv;
   Id max_base = std::numeric_limits<Id>::max();
   Id max_query = std::numeric_limits<Id>::max();
   int repeats = 5;
   int warmup = 1;
   int threads = omp_get_max_threads();
   int query_batch = 2048;
   int group_chunk = 65536;
   int frontier_delta = 2;
   int frontier_cover_cap = 8192;
   int compact_output_cap = 32768;
   std::vector<int> frontier_deltas;
   std::vector<int> frontier_cover_caps;
   bool check = true;
   int gpu = 0;
};

struct LabelTable
{
   std::vector<Id> offsets;
   std::vector<Label> labels;

   Id size() const { return offsets.empty() ? 0 : static_cast<Id>(offsets.size() - 1); }
};

struct GroupTable
{
   LabelTable table;
   std::vector<Id> group_size;
};

struct LabelBitsets
{
   Id num_groups = 0;
   Id words_per_query = 0;
   Id max_group_label_size = 0;
   std::vector<uint64_t> label_group_bits;
   std::vector<uint64_t> size_group_bits;
   std::unordered_map<Label, Id> label_to_dense;
};

struct DescendantBitsets
{
   Id num_groups = 0;
   Id words_per_query = 0;
   std::vector<uint64_t> group_desc_bits;
};

struct BenchResult
{
   std::string provider;
   double total_ms = 0.0;
   double kernel_ms = 0.0;
   double d2h_ms = 0.0;
   double prune_ms = 0.0;
   double avg_frontier_groups = 0.0;
   double avg_candidates = 0.0;
   double avg_min_groups = 0.0;
   double avg_output_groups = 0.0;
   uint64_t output_overflow_queries = 0;
   uint64_t output_truncated_groups = 0;
   uint64_t checksum = 0;
};

double now_ms()
{
   using Clock = std::chrono::high_resolution_clock;
   static const auto t0 = Clock::now();
   return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

std::vector<int> parse_int_list(const std::string &value, const char *name)
{
   std::vector<int> out;
   std::stringstream ss(value);
   std::string item;
   while (std::getline(ss, item, ','))
   {
      if (item.empty())
         continue;
      int v = std::stoi(item);
      if (v <= 0)
         throw std::runtime_error(std::string(name) + " values must be positive");
      out.push_back(v);
   }
   if (out.empty())
      throw std::runtime_error(std::string(name) + " must contain at least one integer");
   std::sort(out.begin(), out.end());
   out.erase(std::unique(out.begin(), out.end()), out.end());
   return out;
}

Options parse_args(int argc, char **argv)
{
   Options opt;
   for (int i = 1; i < argc; ++i)
   {
      std::string a = argv[i];
      auto need = [&](const char *name) -> std::string {
         if (i + 1 >= argc)
            throw std::runtime_error(std::string("missing value for ") + name);
         return argv[++i];
      };
      if (a == "--base-label-file")
         opt.base_label_file = need("--base-label-file");
      else if (a == "--query-label-file")
         opt.query_label_file = need("--query-label-file");
      else if (a == "--provider")
         opt.provider = need("--provider");
      else if (a == "--output-csv")
         opt.output_csv = need("--output-csv");
      else if (a == "--max-base")
         opt.max_base = static_cast<Id>(std::stoul(need("--max-base")));
      else if (a == "--max-query")
         opt.max_query = static_cast<Id>(std::stoul(need("--max-query")));
      else if (a == "--repeats")
         opt.repeats = std::stoi(need("--repeats"));
      else if (a == "--warmup")
         opt.warmup = std::stoi(need("--warmup"));
      else if (a == "--threads")
         opt.threads = std::stoi(need("--threads"));
      else if (a == "--query-batch")
         opt.query_batch = std::stoi(need("--query-batch"));
      else if (a == "--group-chunk")
         opt.group_chunk = std::stoi(need("--group-chunk"));
      else if (a == "--frontier-delta")
         opt.frontier_delta = std::stoi(need("--frontier-delta"));
      else if (a == "--frontier-cover-cap")
         opt.frontier_cover_cap = std::stoi(need("--frontier-cover-cap"));
      else if (a == "--compact-output-cap")
         opt.compact_output_cap = std::stoi(need("--compact-output-cap"));
      else if (a == "--frontier-deltas")
         opt.frontier_deltas = parse_int_list(need("--frontier-deltas"), "--frontier-deltas");
      else if (a == "--frontier-cover-caps")
         opt.frontier_cover_caps = parse_int_list(need("--frontier-cover-caps"), "--frontier-cover-caps");
      else if (a == "--gpu")
         opt.gpu = std::stoi(need("--gpu"));
      else if (a == "--no-check")
         opt.check = false;
      else if (a == "--help" || a == "-h")
      {
         std::cout
             << "Usage: query_entry_group_bench --base-label-file labels.txt --query-label-file query_labels.txt [options]\n"
             << "Providers: cpu_scan, cpu_exact, cpu_cover_frontier, gpu_scan, gpu_bitset, gpu_cover_frontier, gpu_cover_frontier_compact, cover_frontier_sweep, all\n"
             << "  cpu_scan/gpu_scan/gpu_bitset return raw candidate groups for diagnostics.\n"
             << "  cpu_exact returns exact-minimal entry groups and is slow on many-group workloads.\n"
             << "  cpu_cover_frontier is the same coverage-correct semantics as gpu_cover_frontier on CPU OpenMP threads.\n"
             << "  gpu_cover_frontier is the current coverage-correct GPU entry provider.\n"
             << "  gpu_cover_frontier_compact keeps the same GPU coverage path but D2H-copies compact group ids instead of full output bitsets.\n"
             << "Options:\n"
             << "  --provider NAME       Provider to run, default all\n"
             << "  --output-csv PATH     Write one-row-per-provider CSV\n"
             << "  --max-base N          Limit base vectors before grouping\n"
             << "  --max-query N         Limit query count\n"
             << "  --query-batch N       GPU query batch, default 2048\n"
             << "  --group-chunk N       GPU group chunk, default 65536\n"
             << "  --frontier-delta N    Keep size buckets query_len..query_len+N before coverage fallback, default 2\n"
             << "  --frontier-cover-cap N  Max compacted frontier ids used for coverage OR per query, default 8192\n"
             << "  --compact-output-cap N  Max compacted output ids copied per query by gpu_cover_frontier_compact, default 32768\n"
             << "  --frontier-deltas CSV   Sweep deltas for provider=cover_frontier_sweep, default uses --frontier-delta\n"
             << "  --frontier-cover-caps CSV  Sweep caps for provider=cover_frontier_sweep, default uses --frontier-cover-cap\n"
             << "  --repeats N           Timed repeats, default 5\n"
             << "  --warmup N            Warmup repeats, default 1\n"
             << "  --threads N           CPU OpenMP threads\n"
             << "  --gpu N               CUDA device id\n"
             << "  --no-check            Skip comparing GPU output to CPU exact\n";
         std::exit(0);
      }
      else
         throw std::runtime_error("unknown argument: " + a);
   }
   if (opt.base_label_file.empty() || opt.query_label_file.empty())
      throw std::runtime_error("--base-label-file and --query-label-file are required");
   if (opt.query_batch <= 0 || opt.group_chunk <= 0 || opt.repeats <= 0 || opt.warmup < 0 ||
       opt.frontier_delta < 0 || opt.frontier_cover_cap <= 0 || opt.compact_output_cap <= 0)
      throw std::runtime_error("invalid non-positive benchmark parameter");
   if (opt.frontier_deltas.empty())
      opt.frontier_deltas.push_back(opt.frontier_delta);
   if (opt.frontier_cover_caps.empty())
      opt.frontier_cover_caps.push_back(opt.frontier_cover_cap);
   return opt;
}

std::vector<Label> parse_label_line(const std::string &line)
{
   std::vector<Label> labels;
   std::stringstream ss(line);
   std::string item;
   while (std::getline(ss, item, ','))
   {
      if (!item.empty())
         labels.push_back(static_cast<Label>(std::stoul(item)));
   }
   std::sort(labels.begin(), labels.end());
   labels.erase(std::unique(labels.begin(), labels.end()), labels.end());
   return labels;
}

std::vector<std::vector<Label>> read_label_sets(const std::string &path, Id max_rows)
{
   std::ifstream in(path);
   if (!in)
      throw std::runtime_error("failed to open label file: " + path);
   std::vector<std::vector<Label>> rows;
   rows.reserve(std::min<Id>(max_rows, 1u << 20));
   std::string line;
   while (rows.size() < max_rows && std::getline(in, line))
      rows.push_back(parse_label_line(line));
   return rows;
}

LabelTable flatten_rows(const std::vector<std::vector<Label>> &rows)
{
   LabelTable t;
   t.offsets.resize(rows.size() + 1);
   size_t total = 0;
   for (size_t i = 0; i < rows.size(); ++i)
   {
      t.offsets[i] = static_cast<Id>(total);
      total += rows[i].size();
   }
   t.offsets[rows.size()] = static_cast<Id>(total);
   t.labels.reserve(total);
   for (const auto &r : rows)
      t.labels.insert(t.labels.end(), r.begin(), r.end());
   return t;
}

GroupTable build_groups(const std::vector<std::vector<Label>> &base_rows)
{
   struct VecHash
   {
      size_t operator()(const std::vector<Label> &v) const
      {
         size_t h = 1469598103934665603ull;
         for (Label x : v)
         {
            h ^= static_cast<size_t>(x) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
            h *= 1099511628211ull;
         }
         return h;
      }
   };

   std::unordered_map<std::vector<Label>, Id, VecHash> label_to_gid;
   std::vector<std::vector<Label>> group_rows(1);
   std::vector<Id> group_size(1, 0);
   label_to_gid.reserve(base_rows.size());
   for (const auto &labels : base_rows)
   {
      auto it = label_to_gid.find(labels);
      if (it == label_to_gid.end())
      {
         Id gid = static_cast<Id>(group_rows.size());
         label_to_gid.emplace(labels, gid);
         group_rows.push_back(labels);
         group_size.push_back(1);
      }
      else
      {
         group_size[it->second]++;
      }
   }

   GroupTable groups;
   groups.table = flatten_rows(group_rows);
   groups.group_size = std::move(group_size);
   return groups;
}

LabelBitsets build_label_bitsets(const GroupTable &groups)
{
   LabelBitsets bitsets;
   bitsets.num_groups = groups.table.size();
   bitsets.words_per_query = (bitsets.num_groups + 63) / 64;
   for (Id gid = 1; gid < groups.table.size(); ++gid)
   {
      Id len = groups.table.offsets[gid + 1] - groups.table.offsets[gid];
      bitsets.max_group_label_size = std::max(bitsets.max_group_label_size, len);
   }
   for (Id gid = 1; gid < groups.table.size(); ++gid)
   {
      for (Id off = groups.table.offsets[gid]; off < groups.table.offsets[gid + 1]; ++off)
      {
         Label label = groups.table.labels[off];
         if (bitsets.label_to_dense.find(label) == bitsets.label_to_dense.end())
         {
            Id dense = static_cast<Id>(bitsets.label_to_dense.size());
            bitsets.label_to_dense.emplace(label, dense);
         }
      }
   }

   bitsets.label_group_bits.assign(static_cast<size_t>(bitsets.label_to_dense.size()) * bitsets.words_per_query, 0);
   bitsets.size_group_bits.assign(static_cast<size_t>(bitsets.max_group_label_size + 1) * bitsets.words_per_query, 0);
   for (Id gid = 1; gid < groups.table.size(); ++gid)
   {
      const uint64_t bit = uint64_t{1} << (gid & 63);
      const Id word = gid >> 6;
      const Id group_len = groups.table.offsets[gid + 1] - groups.table.offsets[gid];
      bitsets.size_group_bits[static_cast<size_t>(group_len) * bitsets.words_per_query + word] |= bit;
      for (Id off = groups.table.offsets[gid]; off < groups.table.offsets[gid + 1]; ++off)
      {
         Id dense = bitsets.label_to_dense.at(groups.table.labels[off]);
         bitsets.label_group_bits[static_cast<size_t>(dense) * bitsets.words_per_query + word] |= bit;
      }
   }
   return bitsets;
}

DescendantBitsets build_descendant_bitsets_from_labels(const GroupTable &groups, const LabelBitsets &label_bitsets)
{
   DescendantBitsets desc;
   desc.num_groups = groups.table.size();
   desc.words_per_query = label_bitsets.words_per_query;
   desc.group_desc_bits.assign(static_cast<size_t>(desc.num_groups) * desc.words_per_query, 0);

#pragma omp parallel for schedule(dynamic, 32)
   for (int gid_i = 1; gid_i < static_cast<int>(groups.table.size()); ++gid_i)
   {
      Id gid = static_cast<Id>(gid_i);
      uint64_t *dst = desc.group_desc_bits.data() + static_cast<size_t>(gid) * desc.words_per_query;
      std::fill(dst, dst + desc.words_per_query, ~uint64_t{0});
      for (Id off = groups.table.offsets[gid]; off < groups.table.offsets[gid + 1]; ++off)
      {
         auto it = label_bitsets.label_to_dense.find(groups.table.labels[off]);
         if (it == label_bitsets.label_to_dense.end())
         {
            std::fill(dst, dst + desc.words_per_query, 0);
            break;
         }
         const uint64_t *label_bits = label_bitsets.label_group_bits.data() +
                                      static_cast<size_t>(it->second) * desc.words_per_query;
         for (Id word = 0; word < desc.words_per_query; ++word)
            dst[word] &= label_bits[word];
      }
   }
   return desc;
}

LabelTable remap_queries_to_dense(const LabelTable &queries, const std::unordered_map<Label, Id> &label_to_dense)
{
   LabelTable dense = queries;
   for (Label &label : dense.labels)
   {
      auto it = label_to_dense.find(label);
      label = it == label_to_dense.end() ? kInvalidDenseLabel : it->second;
   }
   return dense;
}

bool contains_query(const Label *g, Id g_len, const Label *q, Id q_len)
{
   Id i = 0, j = 0;
   while (i < g_len && j < q_len)
   {
      if (g[i] < q[j])
         ++i;
      else if (g[i] == q[j])
      {
         ++i;
         ++j;
      }
      else
         return false;
   }
   return j == q_len;
}

std::vector<Id> prune_min_super_sets(const GroupTable &groups, std::vector<Id> candidates)
{
   std::sort(candidates.begin(), candidates.end(), [&](Id a, Id b) {
      Id alen = groups.table.offsets[a + 1] - groups.table.offsets[a];
      Id blen = groups.table.offsets[b + 1] - groups.table.offsets[b];
      if (alen != blen)
         return alen < blen;
      return a < b;
   });
   candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());

   std::vector<Id> mins;
   mins.reserve(candidates.size());
   for (Id gid : candidates)
   {
      const Label *g = groups.table.labels.data() + groups.table.offsets[gid];
      Id g_len = groups.table.offsets[gid + 1] - groups.table.offsets[gid];
      bool is_min = true;
      for (Id mid : mins)
      {
         const Label *m = groups.table.labels.data() + groups.table.offsets[mid];
         Id m_len = groups.table.offsets[mid + 1] - groups.table.offsets[mid];
         if (g_len > m_len && contains_query(g, g_len, m, m_len))
         {
            is_min = false;
            break;
         }
      }
      if (is_min)
         mins.push_back(gid);
   }
   return mins;
}

BenchResult run_cpu_exact(const GroupTable &groups, const LabelTable &queries, int threads)
{
   BenchResult r;
   r.provider = "cpu_exact";
   std::vector<uint64_t> candidate_counts(queries.size());
   std::vector<uint64_t> min_counts(queries.size());
   std::vector<uint64_t> checksums(queries.size());
   std::vector<std::vector<Id>> candidates(queries.size());
   omp_set_num_threads(threads);
   double t0 = now_ms();

   double scan0 = now_ms();
#pragma omp parallel for schedule(dynamic, 8)
   for (int qi = 0; qi < static_cast<int>(queries.size()); ++qi)
   {
      const Label *q = queries.labels.data() + queries.offsets[qi];
      Id q_len = queries.offsets[qi + 1] - queries.offsets[qi];
      auto &dst = candidates[qi];
      dst.reserve(128);
      for (Id gid = 1; gid < groups.table.size(); ++gid)
      {
         const Label *g = groups.table.labels.data() + groups.table.offsets[gid];
         Id g_len = groups.table.offsets[gid + 1] - groups.table.offsets[gid];
         if (contains_query(g, g_len, q, q_len))
            dst.push_back(gid);
      }
      candidate_counts[qi] = dst.size();
   }
   r.kernel_ms = now_ms() - scan0;

   double prune0 = now_ms();
#pragma omp parallel for schedule(dynamic, 8)
   for (int qi = 0; qi < static_cast<int>(queries.size()); ++qi)
   {
      auto mins = prune_min_super_sets(groups, std::move(candidates[qi]));
      min_counts[qi] = mins.size();
      uint64_t h = 1469598103934665603ull;
      for (Id gid : mins)
         h = (h ^ gid) * 1099511628211ull;
      checksums[qi] = h;
   }
   r.prune_ms = now_ms() - prune0;

   r.total_ms = now_ms() - t0;
   r.avg_candidates = std::accumulate(candidate_counts.begin(), candidate_counts.end(), 0.0) / queries.size();
   r.avg_min_groups = std::accumulate(min_counts.begin(), min_counts.end(), 0.0) / queries.size();
   r.avg_output_groups = r.avg_min_groups;
   r.checksum = std::accumulate(checksums.begin(), checksums.end(), uint64_t{0}, std::bit_xor<uint64_t>());
   return r;
}

BenchResult run_cpu_scan(const GroupTable &groups, const LabelTable &queries, int threads)
{
   BenchResult r;
   r.provider = "cpu_scan";
   std::vector<uint64_t> candidate_counts(queries.size());
   std::vector<uint64_t> checksums(queries.size());
   std::vector<std::vector<Id>> output_group_ids(queries.size());
   omp_set_num_threads(threads);

   double t0 = now_ms();
#pragma omp parallel for schedule(dynamic, 8)
   for (int qi = 0; qi < static_cast<int>(queries.size()); ++qi)
   {
      const Label *q = queries.labels.data() + queries.offsets[qi];
      Id q_len = queries.offsets[qi + 1] - queries.offsets[qi];
      auto &dst = output_group_ids[qi];
      dst.reserve(128);
      uint64_t h = 1469598103934665603ull;
      for (Id gid = 1; gid < groups.table.size(); ++gid)
      {
         const Label *g = groups.table.labels.data() + groups.table.offsets[gid];
         Id g_len = groups.table.offsets[gid + 1] - groups.table.offsets[gid];
         if (contains_query(g, g_len, q, q_len))
         {
            dst.push_back(gid);
            h = (h ^ gid) * 1099511628211ull;
         }
      }
      candidate_counts[qi] = dst.size();
      checksums[qi] = h;
   }

   r.total_ms = now_ms() - t0;
   r.kernel_ms = r.total_ms;
   r.avg_candidates = std::accumulate(candidate_counts.begin(), candidate_counts.end(), 0.0) / queries.size();
   r.avg_min_groups = r.avg_candidates;
   r.avg_output_groups = r.avg_min_groups;
   r.checksum = std::accumulate(checksums.begin(), checksums.end(), uint64_t{0}, std::bit_xor<uint64_t>());
   return r;
}

__device__ bool contains_query_device(const Label *g_labels, Id g_begin, Id g_end,
                                      const Label *q_labels, Id q_begin, Id q_end)
{
   Id i = g_begin;
   Id j = q_begin;
   while (i < g_end && j < q_end)
   {
      Label gv = g_labels[i];
      Label qv = q_labels[j];
      if (gv < qv)
         ++i;
      else if (gv == qv)
      {
         ++i;
         ++j;
      }
      else
         return false;
   }
   return j == q_end;
}

__global__ void containment_scan_kernel(const Label *group_labels, const Id *group_offsets,
                                        const Label *query_labels, const Id *query_offsets,
                                        Id group_begin, Id num_groups_chunk,
                                        Id query_begin, Id num_queries_chunk,
                                        uint8_t *flags)
{
   Id idx = blockIdx.x * blockDim.x + threadIdx.x;
   Id total = num_groups_chunk * num_queries_chunk;
   if (idx >= total)
      return;
   Id local_g = idx % num_groups_chunk;
   Id local_q = idx / num_groups_chunk;
   Id gid = group_begin + local_g;
   Id qid = query_begin + local_q;
   bool ok = contains_query_device(group_labels, group_offsets[gid], group_offsets[gid + 1],
                                   query_labels, query_offsets[qid], query_offsets[qid + 1]);
   flags[idx] = ok ? 1 : 0;
}

class DeviceTables
{
public:
   DeviceTables(const GroupTable &groups, const LabelTable &queries)
   {
      CUDA_CHECK(cudaMalloc(&d_group_offsets_, groups.table.offsets.size() * sizeof(Id)));
      CUDA_CHECK(cudaMalloc(&d_group_labels_, groups.table.labels.size() * sizeof(Label)));
      CUDA_CHECK(cudaMalloc(&d_query_offsets_, queries.offsets.size() * sizeof(Id)));
      CUDA_CHECK(cudaMalloc(&d_query_labels_, queries.labels.size() * sizeof(Label)));
      CUDA_CHECK(cudaMemcpy(d_group_offsets_, groups.table.offsets.data(), groups.table.offsets.size() * sizeof(Id), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_group_labels_, groups.table.labels.data(), groups.table.labels.size() * sizeof(Label), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_query_offsets_, queries.offsets.data(), queries.offsets.size() * sizeof(Id), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_query_labels_, queries.labels.data(), queries.labels.size() * sizeof(Label), cudaMemcpyHostToDevice));
   }

   ~DeviceTables()
   {
      cudaFree(d_group_offsets_);
      cudaFree(d_group_labels_);
      cudaFree(d_query_offsets_);
      cudaFree(d_query_labels_);
   }

   const Id *group_offsets() const { return d_group_offsets_; }
   const Label *group_labels() const { return d_group_labels_; }
   const Id *query_offsets() const { return d_query_offsets_; }
   const Label *query_labels() const { return d_query_labels_; }

private:
   Id *d_group_offsets_ = nullptr;
   Label *d_group_labels_ = nullptr;
   Id *d_query_offsets_ = nullptr;
   Label *d_query_labels_ = nullptr;
};

class DeviceBitsetTables
{
public:
   DeviceBitsetTables(const LabelBitsets &bitsets, const LabelTable &dense_queries)
   {
      CUDA_CHECK(cudaMalloc(&d_label_group_bits_, bitsets.label_group_bits.size() * sizeof(uint64_t)));
      CUDA_CHECK(cudaMalloc(&d_size_group_bits_, bitsets.size_group_bits.size() * sizeof(uint64_t)));
      CUDA_CHECK(cudaMalloc(&d_query_offsets_, dense_queries.offsets.size() * sizeof(Id)));
      CUDA_CHECK(cudaMalloc(&d_query_dense_labels_, dense_queries.labels.size() * sizeof(Label)));
      CUDA_CHECK(cudaMemcpy(d_label_group_bits_, bitsets.label_group_bits.data(),
                            bitsets.label_group_bits.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_size_group_bits_, bitsets.size_group_bits.data(),
                            bitsets.size_group_bits.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_query_offsets_, dense_queries.offsets.data(),
                            dense_queries.offsets.size() * sizeof(Id), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_query_dense_labels_, dense_queries.labels.data(),
                            dense_queries.labels.size() * sizeof(Label), cudaMemcpyHostToDevice));
   }

   ~DeviceBitsetTables()
   {
      cudaFree(d_label_group_bits_);
      cudaFree(d_size_group_bits_);
      cudaFree(d_query_offsets_);
      cudaFree(d_query_dense_labels_);
   }

   const uint64_t *label_group_bits() const { return d_label_group_bits_; }
   const uint64_t *size_group_bits() const { return d_size_group_bits_; }
   const Id *query_offsets() const { return d_query_offsets_; }
   const Label *query_dense_labels() const { return d_query_dense_labels_; }

private:
   uint64_t *d_label_group_bits_ = nullptr;
   uint64_t *d_size_group_bits_ = nullptr;
   Id *d_query_offsets_ = nullptr;
   Label *d_query_dense_labels_ = nullptr;
};

class DeviceDescendantBitsets
{
public:
   explicit DeviceDescendantBitsets(const DescendantBitsets &desc)
       : num_groups_(desc.num_groups)
   {
      CUDA_CHECK(cudaMalloc(&d_group_desc_bits_, desc.group_desc_bits.size() * sizeof(uint64_t)));
      CUDA_CHECK(cudaMemcpy(d_group_desc_bits_, desc.group_desc_bits.data(),
                            desc.group_desc_bits.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
   }

   ~DeviceDescendantBitsets()
   {
      cudaFree(d_group_desc_bits_);
   }

   const uint64_t *group_desc_bits() const { return d_group_desc_bits_; }
   Id num_groups() const { return num_groups_; }

private:
   uint64_t *d_group_desc_bits_ = nullptr;
   Id num_groups_ = 0;
};

__global__ void bitset_intersection_kernel(const uint64_t *label_group_bits,
                                           Id words_per_query,
                                           const Label *query_dense_labels,
                                           const Id *query_offsets,
                                           Id query_begin,
                                           Id num_queries_chunk,
                                           uint64_t *out_bits)
{
   Id word = blockIdx.x * blockDim.x + threadIdx.x;
   Id local_q = blockIdx.y;
   if (local_q >= num_queries_chunk || word >= words_per_query)
      return;

   Id qid = query_begin + local_q;
   Id q_begin = query_offsets[qid];
   Id q_end = query_offsets[qid + 1];
   uint64_t mask = ~uint64_t{0};
   for (Id off = q_begin; off < q_end; ++off)
   {
      Label dense = query_dense_labels[off];
      if (dense == kInvalidDenseLabel)
      {
         mask = 0;
         break;
      }
      mask &= label_group_bits[static_cast<size_t>(dense) * words_per_query + word];
   }
   out_bits[static_cast<size_t>(local_q) * words_per_query + word] = mask;
}

__global__ void candidate_frontier_kernel(const uint64_t *label_group_bits,
                                          const uint64_t *size_group_bits,
                                          Id words_per_query,
                                          Id max_group_label_size,
                                          const Label *query_dense_labels,
                                          const Id *query_offsets,
                                          Id query_begin,
                                          Id num_queries_chunk,
                                          Id frontier_delta,
                                          uint64_t *candidate_bits,
                                          uint64_t *frontier_bits)
{
   Id word = blockIdx.x * blockDim.x + threadIdx.x;
   Id local_q = blockIdx.y;
   if (local_q >= num_queries_chunk || word >= words_per_query)
      return;

   Id qid = query_begin + local_q;
   Id q_begin = query_offsets[qid];
   Id q_end = query_offsets[qid + 1];
   uint64_t candidates = ~uint64_t{0};
   for (Id off = q_begin; off < q_end; ++off)
   {
      Label dense = query_dense_labels[off];
      if (dense == kInvalidDenseLabel)
      {
         candidates = 0;
         break;
      }
      candidates &= label_group_bits[static_cast<size_t>(dense) * words_per_query + word];
   }

   Id q_len = q_end - q_begin;
   Id end_len = min(max_group_label_size, q_len + frontier_delta);
   uint64_t size_mask = 0;
   for (Id len = q_len; len <= end_len; ++len)
      size_mask |= size_group_bits[static_cast<size_t>(len) * words_per_query + word];

   const size_t out = static_cast<size_t>(local_q) * words_per_query + word;
   candidate_bits[out] = candidates;
   frontier_bits[out] = candidates & size_mask;
}

__global__ void compact_frontier_ids_kernel(const uint64_t *frontier_bits,
                                            Id num_groups,
                                            Id words_per_query,
                                            Id num_queries_chunk,
                                            Id frontier_cover_cap,
                                            uint64_t *selected_frontier_bits,
                                            Id *frontier_counts,
                                            Id *frontier_ids)
{
   Id local_q = blockIdx.x;
   if (local_q >= num_queries_chunk)
      return;

   const uint64_t *frontier = frontier_bits + static_cast<size_t>(local_q) * words_per_query;
   __shared__ Id count;
   if (threadIdx.x == 0)
      count = 0;
   __syncthreads();

   for (Id fword = threadIdx.x; fword < words_per_query; fword += blockDim.x)
   {
      uint64_t bits = frontier[fword];
      while (bits)
      {
         Id bit = static_cast<Id>(__ffsll(static_cast<long long>(bits)) - 1);
         Id gid = (fword << 6) + bit;
         if (gid > 0 && gid < num_groups)
         {
            Id slot = count;
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

__global__ void descendant_cover_list_kernel(const Id *frontier_counts,
                                             const Id *frontier_ids,
                                             const uint64_t *group_desc_bits,
                                             Id words_per_query,
                                             Id num_queries_chunk,
                                             Id frontier_cover_cap,
                                             uint64_t *covered_bits)
{
   Id word = blockIdx.x * blockDim.x + threadIdx.x;
   Id local_q = blockIdx.y;
   if (local_q >= num_queries_chunk || word >= words_per_query)
      return;

   Id count = frontier_counts[local_q];
   const Id *ids = frontier_ids + static_cast<size_t>(local_q) * frontier_cover_cap;
   uint64_t covered = 0;
   for (Id i = 0; i < count; ++i)
   {
      Id gid = ids[i];
      covered |= group_desc_bits[static_cast<size_t>(gid) * words_per_query + word];
   }
   covered_bits[static_cast<size_t>(local_q) * words_per_query + word] = covered;
}

__global__ void cover_frontier_select_kernel(const uint64_t *candidate_bits,
                                             const uint64_t *selected_frontier_bits,
                                             const uint64_t *covered_bits,
                                             Id words_per_query,
                                             Id num_queries_chunk,
                                             uint64_t *out_bits)
{
   Id word = blockIdx.x * blockDim.x + threadIdx.x;
   Id local_q = blockIdx.y;
   if (local_q >= num_queries_chunk || word >= words_per_query)
      return;

   const size_t idx = static_cast<size_t>(local_q) * words_per_query + word;
   out_bits[idx] = selected_frontier_bits[idx] | (candidate_bits[idx] & ~covered_bits[idx]);
}

__global__ void cover_frontier_select_count_kernel(const uint64_t *candidate_bits,
                                                   const uint64_t *selected_frontier_bits,
                                                   const uint64_t *covered_bits,
                                                   Id num_groups,
                                                   Id words_per_query,
                                                   Id num_queries_chunk,
                                                   uint64_t *out_bits,
                                                   Id *output_counts)
{
   Id word = blockIdx.x * blockDim.x + threadIdx.x;
   Id local_q = blockIdx.y;
   if (local_q >= num_queries_chunk)
      return;

   const size_t idx = static_cast<size_t>(local_q) * words_per_query + word;
   uint64_t out = 0;
   if (word < words_per_query)
   {
      out = selected_frontier_bits[idx] | (candidate_bits[idx] & ~covered_bits[idx]);
      out_bits[idx] = out;
   }

   Id first_gid = word << 6;
   if (word >= words_per_query)
      out = 0;
   else if (first_gid == 0)
      out &= ~uint64_t{1};
   if (first_gid + 64 > num_groups)
   {
      Id valid = num_groups > first_gid ? num_groups - first_gid : 0;
      out &= valid >= 64 ? ~uint64_t{0} : ((uint64_t{1} << valid) - 1);
   }

   Id count = static_cast<Id>(__popcll(static_cast<unsigned long long>(out)));
   __shared__ Id partial[256];
   partial[threadIdx.x] = count;
   __syncthreads();
   for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
   {
      if (threadIdx.x < stride)
         partial[threadIdx.x] += partial[threadIdx.x + stride];
      __syncthreads();
   }
   if (threadIdx.x == 0 && partial[0] != 0)
      atomicAdd(output_counts + local_q, partial[0]);
}

__global__ void compact_output_ids_kernel(const uint64_t *out_bits,
                                          Id num_groups,
                                          Id words_per_query,
                                          Id num_queries_chunk,
                                          Id compact_output_cap,
                                          const Id *output_offsets,
                                          Id *output_ids)
{
   Id local_q = blockIdx.x;
   if (local_q >= num_queries_chunk)
      return;

   const uint64_t *row = out_bits + static_cast<size_t>(local_q) * words_per_query;
   __shared__ Id written;
   if (threadIdx.x == 0)
      written = 0;
   __syncthreads();

   Id base = output_offsets[local_q];
   for (Id word = threadIdx.x; word < words_per_query; word += blockDim.x)
   {
      uint64_t bits = row[word];
      Id first_gid = word << 6;
      if (first_gid == 0)
         bits &= ~uint64_t{1};
      if (first_gid + 64 > num_groups)
      {
         Id valid = num_groups > first_gid ? num_groups - first_gid : 0;
         bits &= valid >= 64 ? ~uint64_t{0} : ((uint64_t{1} << valid) - 1);
      }
      while (bits)
      {
         Id bit = static_cast<Id>(__ffsll(static_cast<long long>(bits)) - 1);
         Id slot = atomicAdd(&written, 1u);
         if (slot < compact_output_cap)
            output_ids[static_cast<size_t>(base) + slot] = first_gid + bit;
         bits &= bits - 1;
      }
   }
}

BenchResult run_gpu_scan_once(const GroupTable &groups, const LabelTable &queries,
                              const DeviceTables &dev, int query_batch, int group_chunk)
{
   BenchResult r;
   r.provider = "gpu_scan";
   std::vector<uint64_t> candidate_counts(queries.size(), 0);
   std::vector<uint64_t> min_counts(queries.size(), 0);
   std::vector<uint64_t> checksums(queries.size(), 0);

   const Id num_groups = groups.table.size();
   const Id num_queries = queries.size();
   const size_t max_flags = static_cast<size_t>(query_batch) * static_cast<size_t>(group_chunk);
   uint8_t *d_flags = nullptr;
   CUDA_CHECK(cudaMalloc(&d_flags, max_flags));
   std::vector<uint8_t> h_flags(max_flags);
   std::vector<std::vector<Id>> candidates(num_queries);

   cudaEvent_t ev0, ev1;
   CUDA_CHECK(cudaEventCreate(&ev0));
   CUDA_CHECK(cudaEventCreate(&ev1));

   double t0 = now_ms();
   for (Id qb = 0; qb < num_queries; qb += query_batch)
   {
      Id qn = std::min<Id>(query_batch, num_queries - qb);
      for (Id gb = 1; gb < num_groups; gb += group_chunk)
      {
         Id gn = std::min<Id>(group_chunk, num_groups - gb);
         int threads = 256;
         int blocks = static_cast<int>((static_cast<uint64_t>(qn) * gn + threads - 1) / threads);
         CUDA_CHECK(cudaEventRecord(ev0));
         containment_scan_kernel<<<blocks, threads>>>(dev.group_labels(), dev.group_offsets(),
                                                      dev.query_labels(), dev.query_offsets(),
                                                      gb, gn, qb, qn, d_flags);
         CUDA_CHECK(cudaEventRecord(ev1));
         CUDA_CHECK(cudaEventSynchronize(ev1));
         float kernel_ms = 0.0f;
         CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, ev0, ev1));
         r.kernel_ms += kernel_ms;

         double d2h0 = now_ms();
         CUDA_CHECK(cudaMemcpy(h_flags.data(), d_flags, static_cast<size_t>(qn) * gn, cudaMemcpyDeviceToHost));
         r.d2h_ms += now_ms() - d2h0;

         for (Id lq = 0; lq < qn; ++lq)
         {
            auto &dst = candidates[qb + lq];
            const uint8_t *row = h_flags.data() + static_cast<size_t>(lq) * gn;
            for (Id lg = 0; lg < gn; ++lg)
               if (row[lg])
                  dst.push_back(gb + lg);
         }
      }
   }

   double prune0 = now_ms();
#pragma omp parallel for schedule(dynamic, 8)
   for (int qi = 0; qi < static_cast<int>(num_queries); ++qi)
   {
      candidate_counts[qi] = candidates[qi].size();
      auto mins = prune_min_super_sets(groups, std::move(candidates[qi]));
      min_counts[qi] = mins.size();
      uint64_t h = 1469598103934665603ull;
      for (Id gid : mins)
         h = (h ^ gid) * 1099511628211ull;
      checksums[qi] = h;
   }
   r.prune_ms = now_ms() - prune0;
   CUDA_CHECK(cudaFree(d_flags));
   CUDA_CHECK(cudaEventDestroy(ev0));
   CUDA_CHECK(cudaEventDestroy(ev1));
   r.total_ms = now_ms() - t0;
   r.avg_candidates = std::accumulate(candidate_counts.begin(), candidate_counts.end(), 0.0) / queries.size();
   r.avg_min_groups = std::accumulate(min_counts.begin(), min_counts.end(), 0.0) / queries.size();
   r.avg_output_groups = r.avg_min_groups;
   r.checksum = std::accumulate(checksums.begin(), checksums.end(), uint64_t{0}, std::bit_xor<uint64_t>());
   return r;
}

BenchResult run_gpu_bitset_once(const GroupTable &groups, const LabelTable &queries,
                                const LabelBitsets &bitsets, const DeviceBitsetTables &dev,
                                int query_batch)
{
   BenchResult r;
   r.provider = "gpu_bitset";
   std::vector<uint64_t> candidate_counts(queries.size(), 0);
   std::vector<uint64_t> min_counts(queries.size(), 0);
   std::vector<uint64_t> checksums(queries.size(), 0);

   const Id num_queries = queries.size();
   const Id words = bitsets.words_per_query;
   uint64_t *d_out_bits = nullptr;
   CUDA_CHECK(cudaMalloc(&d_out_bits, static_cast<size_t>(query_batch) * words * sizeof(uint64_t)));
   std::vector<uint64_t> h_bits(static_cast<size_t>(query_batch) * words);
   std::vector<std::vector<Id>> candidates(num_queries);

   cudaEvent_t ev0, ev1;
   CUDA_CHECK(cudaEventCreate(&ev0));
   CUDA_CHECK(cudaEventCreate(&ev1));

   double t0 = now_ms();
   for (Id qb = 0; qb < num_queries; qb += query_batch)
   {
      Id qn = std::min<Id>(query_batch, num_queries - qb);
      dim3 block(256);
      dim3 grid((words + block.x - 1) / block.x, qn);
      CUDA_CHECK(cudaEventRecord(ev0));
      bitset_intersection_kernel<<<grid, block>>>(dev.label_group_bits(), words,
                                                  dev.query_dense_labels(), dev.query_offsets(),
                                                  qb, qn, d_out_bits);
      CUDA_CHECK(cudaEventRecord(ev1));
      CUDA_CHECK(cudaEventSynchronize(ev1));
      float kernel_ms = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, ev0, ev1));
      r.kernel_ms += kernel_ms;

      double d2h0 = now_ms();
      CUDA_CHECK(cudaMemcpy(h_bits.data(), d_out_bits, static_cast<size_t>(qn) * words * sizeof(uint64_t), cudaMemcpyDeviceToHost));
      r.d2h_ms += now_ms() - d2h0;

      for (Id lq = 0; lq < qn; ++lq)
      {
         auto &dst = candidates[qb + lq];
         const uint64_t *row = h_bits.data() + static_cast<size_t>(lq) * words;
         for (Id word = 0; word < words; ++word)
         {
            uint64_t bits = row[word];
            while (bits)
            {
               Id bit = static_cast<Id>(__builtin_ctzll(bits));
               Id gid = (word << 6) + bit;
               if (gid > 0 && gid < groups.table.size())
                  dst.push_back(gid);
               bits &= bits - 1;
            }
         }
      }
   }

   double prune0 = now_ms();
#pragma omp parallel for schedule(dynamic, 8)
   for (int qi = 0; qi < static_cast<int>(num_queries); ++qi)
   {
      candidate_counts[qi] = candidates[qi].size();
      auto mins = prune_min_super_sets(groups, std::move(candidates[qi]));
      min_counts[qi] = mins.size();
      uint64_t h = 1469598103934665603ull;
      for (Id gid : mins)
         h = (h ^ gid) * 1099511628211ull;
      checksums[qi] = h;
   }
   r.prune_ms = now_ms() - prune0;
   CUDA_CHECK(cudaFree(d_out_bits));
   CUDA_CHECK(cudaEventDestroy(ev0));
   CUDA_CHECK(cudaEventDestroy(ev1));
   r.total_ms = now_ms() - t0;
   r.avg_candidates = std::accumulate(candidate_counts.begin(), candidate_counts.end(), 0.0) / queries.size();
   r.avg_min_groups = std::accumulate(min_counts.begin(), min_counts.end(), 0.0) / queries.size();
   r.avg_output_groups = r.avg_min_groups;
   r.checksum = std::accumulate(checksums.begin(), checksums.end(), uint64_t{0}, std::bit_xor<uint64_t>());
   return r;
}

BenchResult run_gpu_cover_frontier_once(const GroupTable &groups, const LabelTable &queries,
                                        const LabelBitsets &bitsets, const DeviceBitsetTables &dev,
                                        const DeviceDescendantBitsets &dev_desc,
                                        int query_batch, int frontier_delta, int frontier_cover_cap)
{
   BenchResult r;
   r.provider = "gpu_cover_frontier_d" + std::to_string(frontier_delta) +
                "_cap" + std::to_string(frontier_cover_cap);
   std::vector<uint64_t> output_counts(queries.size(), 0);
   std::vector<uint64_t> checksums(queries.size(), 0);
   std::vector<std::vector<Id>> output_group_ids(queries.size());

   const Id num_queries = queries.size();
   const Id words = bitsets.words_per_query;
   const size_t batch_words = static_cast<size_t>(query_batch) * words;
   uint64_t *d_candidate_bits = nullptr;
   uint64_t *d_frontier_bits = nullptr;
   uint64_t *d_selected_frontier_bits = nullptr;
   uint64_t *d_covered_bits = nullptr;
   uint64_t *d_out_bits = nullptr;
   Id *d_frontier_counts = nullptr;
   Id *d_frontier_ids = nullptr;
   CUDA_CHECK(cudaMalloc(&d_candidate_bits, batch_words * sizeof(uint64_t)));
   CUDA_CHECK(cudaMalloc(&d_frontier_bits, batch_words * sizeof(uint64_t)));
   CUDA_CHECK(cudaMalloc(&d_selected_frontier_bits, batch_words * sizeof(uint64_t)));
   CUDA_CHECK(cudaMalloc(&d_covered_bits, batch_words * sizeof(uint64_t)));
   CUDA_CHECK(cudaMalloc(&d_out_bits, batch_words * sizeof(uint64_t)));
   CUDA_CHECK(cudaMalloc(&d_frontier_counts, static_cast<size_t>(query_batch) * sizeof(Id)));
   CUDA_CHECK(cudaMalloc(&d_frontier_ids, static_cast<size_t>(query_batch) *
                                             static_cast<size_t>(frontier_cover_cap) * sizeof(Id)));
   uint64_t *h_bits = nullptr;
   CUDA_CHECK(cudaMallocHost(&h_bits, batch_words * sizeof(uint64_t)));
   std::vector<Id> h_frontier_counts(query_batch);
   uint64_t frontier_count_sum = 0;

   cudaEvent_t ev0, ev1;
   CUDA_CHECK(cudaEventCreate(&ev0));
   CUDA_CHECK(cudaEventCreate(&ev1));

   double t0 = now_ms();
   for (Id qb = 0; qb < num_queries; qb += query_batch)
   {
      Id qn = std::min<Id>(query_batch, num_queries - qb);
      dim3 block(256);
      dim3 grid((words + block.x - 1) / block.x, qn);
      CUDA_CHECK(cudaEventRecord(ev0));
      candidate_frontier_kernel<<<grid, block>>>(dev.label_group_bits(), dev.size_group_bits(),
                                                 words, bitsets.max_group_label_size,
                                                 dev.query_dense_labels(), dev.query_offsets(),
                                                 qb, qn, static_cast<Id>(frontier_delta),
                                                 d_candidate_bits, d_frontier_bits);
      CUDA_CHECK(cudaMemset(d_selected_frontier_bits, 0, static_cast<size_t>(qn) * words * sizeof(uint64_t)));
      compact_frontier_ids_kernel<<<qn, 256>>>(d_frontier_bits, dev_desc.num_groups(), words, qn,
                                               static_cast<Id>(frontier_cover_cap),
                                               d_selected_frontier_bits,
                                               d_frontier_counts, d_frontier_ids);
      descendant_cover_list_kernel<<<grid, block>>>(d_frontier_counts, d_frontier_ids,
                                                    dev_desc.group_desc_bits(), words, qn,
                                                    static_cast<Id>(frontier_cover_cap),
                                                    d_covered_bits);
      cover_frontier_select_kernel<<<grid, block>>>(d_candidate_bits, d_selected_frontier_bits, d_covered_bits,
                                                    words, qn, d_out_bits);
      CUDA_CHECK(cudaEventRecord(ev1));
      CUDA_CHECK(cudaEventSynchronize(ev1));
      float kernel_ms = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, ev0, ev1));
      r.kernel_ms += kernel_ms;

      double d2h0 = now_ms();
      CUDA_CHECK(cudaMemcpy(h_bits, d_out_bits, static_cast<size_t>(qn) * words * sizeof(uint64_t), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(h_frontier_counts.data(), d_frontier_counts, static_cast<size_t>(qn) * sizeof(Id), cudaMemcpyDeviceToHost));
      r.d2h_ms += now_ms() - d2h0;
      frontier_count_sum += std::accumulate(h_frontier_counts.begin(), h_frontier_counts.begin() + qn, uint64_t{0});

      double mat0 = now_ms();
#pragma omp parallel for schedule(static)
      for (int lq_i = 0; lq_i < static_cast<int>(qn); ++lq_i)
      {
         Id lq = static_cast<Id>(lq_i);
         const uint64_t *row = h_bits + static_cast<size_t>(lq) * words;
         auto &dst = output_group_ids[qb + lq];
         dst.clear();
         uint64_t h = 1469598103934665603ull;
         for (Id word = 0; word < words; ++word)
         {
            uint64_t bits = row[word];
            while (bits)
            {
               Id bit = static_cast<Id>(__builtin_ctzll(bits));
               Id gid = (word << 6) + bit;
               if (gid > 0 && gid < groups.table.size())
               {
                  dst.push_back(gid);
                  h = (h ^ gid) * 1099511628211ull;
               }
               bits &= bits - 1;
            }
         }
         output_counts[qb + lq] = dst.size();
         checksums[qb + lq] = h;
      }
      r.prune_ms += now_ms() - mat0;
   }

   r.total_ms = now_ms() - t0;
   CUDA_CHECK(cudaFree(d_candidate_bits));
   CUDA_CHECK(cudaFree(d_frontier_bits));
   CUDA_CHECK(cudaFree(d_selected_frontier_bits));
   CUDA_CHECK(cudaFree(d_covered_bits));
   CUDA_CHECK(cudaFree(d_out_bits));
   CUDA_CHECK(cudaFree(d_frontier_counts));
   CUDA_CHECK(cudaFree(d_frontier_ids));
   CUDA_CHECK(cudaFreeHost(h_bits));
   CUDA_CHECK(cudaEventDestroy(ev0));
   CUDA_CHECK(cudaEventDestroy(ev1));
   r.avg_frontier_groups = static_cast<double>(frontier_count_sum) / queries.size();
   r.avg_candidates = std::accumulate(output_counts.begin(), output_counts.end(), 0.0) / queries.size();
   r.avg_min_groups = r.avg_candidates;
   r.avg_output_groups = r.avg_min_groups;
   r.checksum = std::accumulate(checksums.begin(), checksums.end(), uint64_t{0}, std::bit_xor<uint64_t>());
   return r;
}

BenchResult run_gpu_cover_frontier_compact_once(const GroupTable &groups, const LabelTable &queries,
                                                const LabelBitsets &bitsets, const DeviceBitsetTables &dev,
                                                const DeviceDescendantBitsets &dev_desc,
                                                int query_batch, int frontier_delta, int frontier_cover_cap,
                                                int compact_output_cap, bool compute_checksum)
{
   BenchResult r;
   r.provider = "gpu_cover_frontier_compact_d" + std::to_string(frontier_delta) +
                "_cap" + std::to_string(frontier_cover_cap) +
                "_outcap" + std::to_string(compact_output_cap);
   std::vector<uint64_t> output_counts(queries.size(), 0);
   std::vector<uint64_t> checksums(queries.size(), 0);

   const Id num_queries = queries.size();
   const Id words = bitsets.words_per_query;
   const size_t batch_words = static_cast<size_t>(query_batch) * words;
   const size_t batch_output_cap = static_cast<size_t>(query_batch) * static_cast<size_t>(compact_output_cap);
   uint64_t *d_candidate_bits = nullptr;
   uint64_t *d_frontier_bits = nullptr;
   uint64_t *d_selected_frontier_bits = nullptr;
   uint64_t *d_covered_bits = nullptr;
   uint64_t *d_out_bits = nullptr;
   Id *d_frontier_counts = nullptr;
   Id *d_frontier_ids = nullptr;
   Id *d_output_counts = nullptr;
   Id *d_output_offsets = nullptr;
   Id *d_output_ids = nullptr;
   CUDA_CHECK(cudaMalloc(&d_candidate_bits, batch_words * sizeof(uint64_t)));
   CUDA_CHECK(cudaMalloc(&d_frontier_bits, batch_words * sizeof(uint64_t)));
   CUDA_CHECK(cudaMalloc(&d_selected_frontier_bits, batch_words * sizeof(uint64_t)));
   CUDA_CHECK(cudaMalloc(&d_covered_bits, batch_words * sizeof(uint64_t)));
   CUDA_CHECK(cudaMalloc(&d_out_bits, batch_words * sizeof(uint64_t)));
   CUDA_CHECK(cudaMalloc(&d_frontier_counts, static_cast<size_t>(query_batch) * sizeof(Id)));
   CUDA_CHECK(cudaMalloc(&d_frontier_ids, static_cast<size_t>(query_batch) *
                                             static_cast<size_t>(frontier_cover_cap) * sizeof(Id)));
   CUDA_CHECK(cudaMalloc(&d_output_counts, static_cast<size_t>(query_batch) * sizeof(Id)));
   CUDA_CHECK(cudaMalloc(&d_output_offsets, static_cast<size_t>(query_batch) * sizeof(Id)));
   CUDA_CHECK(cudaMalloc(&d_output_ids, batch_output_cap * sizeof(Id)));

   Id *h_output_counts = nullptr;
   Id *h_output_offsets = nullptr;
   Id *h_output_ids = nullptr;
   Id *h_frontier_counts = nullptr;
   CUDA_CHECK(cudaMallocHost(&h_output_counts, static_cast<size_t>(query_batch) * sizeof(Id)));
   CUDA_CHECK(cudaMallocHost(&h_output_offsets, static_cast<size_t>(query_batch) * sizeof(Id)));
   CUDA_CHECK(cudaMallocHost(&h_output_ids, batch_output_cap * sizeof(Id)));
   CUDA_CHECK(cudaMallocHost(&h_frontier_counts, static_cast<size_t>(query_batch) * sizeof(Id)));
   uint64_t frontier_count_sum = 0;

   cudaEvent_t ev0, ev1;
   CUDA_CHECK(cudaEventCreate(&ev0));
   CUDA_CHECK(cudaEventCreate(&ev1));

   double t0 = now_ms();
   for (Id qb = 0; qb < num_queries; qb += query_batch)
   {
      Id qn = std::min<Id>(query_batch, num_queries - qb);
      dim3 block(256);
      dim3 grid((words + block.x - 1) / block.x, qn);
      CUDA_CHECK(cudaEventRecord(ev0));
      candidate_frontier_kernel<<<grid, block>>>(dev.label_group_bits(), dev.size_group_bits(),
                                                 words, bitsets.max_group_label_size,
                                                 dev.query_dense_labels(), dev.query_offsets(),
                                                 qb, qn, static_cast<Id>(frontier_delta),
                                                 d_candidate_bits, d_frontier_bits);
      CUDA_CHECK(cudaMemset(d_selected_frontier_bits, 0, static_cast<size_t>(qn) * words * sizeof(uint64_t)));
      compact_frontier_ids_kernel<<<qn, 256>>>(d_frontier_bits, dev_desc.num_groups(), words, qn,
                                               static_cast<Id>(frontier_cover_cap),
                                               d_selected_frontier_bits,
                                               d_frontier_counts, d_frontier_ids);
      descendant_cover_list_kernel<<<grid, block>>>(d_frontier_counts, d_frontier_ids,
                                                    dev_desc.group_desc_bits(), words, qn,
                                                    static_cast<Id>(frontier_cover_cap),
                                                    d_covered_bits);
      CUDA_CHECK(cudaMemset(d_output_counts, 0, static_cast<size_t>(qn) * sizeof(Id)));
      cover_frontier_select_count_kernel<<<grid, block>>>(d_candidate_bits, d_selected_frontier_bits, d_covered_bits,
                                                          groups.table.size(), words, qn,
                                                          d_out_bits, d_output_counts);
      CUDA_CHECK(cudaEventRecord(ev1));
      CUDA_CHECK(cudaEventSynchronize(ev1));
      float kernel_ms = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, ev0, ev1));
      r.kernel_ms += kernel_ms;

      double d2h0 = now_ms();
      CUDA_CHECK(cudaMemcpy(h_output_counts, d_output_counts, static_cast<size_t>(qn) * sizeof(Id), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(h_frontier_counts, d_frontier_counts, static_cast<size_t>(qn) * sizeof(Id), cudaMemcpyDeviceToHost));
      r.d2h_ms += now_ms() - d2h0;
      frontier_count_sum += std::accumulate(h_frontier_counts, h_frontier_counts + qn, uint64_t{0});

      double mat0 = now_ms();
      size_t total_capped_ids = 0;
      for (Id lq = 0; lq < qn; ++lq)
      {
         Id count = h_output_counts[lq];
         output_counts[qb + lq] = count;
         h_output_offsets[lq] = static_cast<Id>(total_capped_ids);
         Id capped = std::min<Id>(count, static_cast<Id>(compact_output_cap));
         total_capped_ids += capped;
         if (count > static_cast<Id>(compact_output_cap))
         {
            ++r.output_overflow_queries;
            r.output_truncated_groups += static_cast<uint64_t>(count - static_cast<Id>(compact_output_cap));
         }
      }
      r.prune_ms += now_ms() - mat0;

      mat0 = now_ms();
      CUDA_CHECK(cudaMemcpy(d_output_offsets, h_output_offsets, static_cast<size_t>(qn) * sizeof(Id), cudaMemcpyHostToDevice));
      r.prune_ms += now_ms() - mat0;

      CUDA_CHECK(cudaEventRecord(ev0));
      compact_output_ids_kernel<<<qn, 256>>>(d_out_bits, groups.table.size(), words, qn,
                                             static_cast<Id>(compact_output_cap),
                                             d_output_offsets, d_output_ids);
      CUDA_CHECK(cudaEventRecord(ev1));
      CUDA_CHECK(cudaEventSynchronize(ev1));
      CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, ev0, ev1));
      r.kernel_ms += kernel_ms;

      d2h0 = now_ms();
      if (total_capped_ids > 0)
         CUDA_CHECK(cudaMemcpy(h_output_ids, d_output_ids, total_capped_ids * sizeof(Id), cudaMemcpyDeviceToHost));
      r.d2h_ms += now_ms() - d2h0;

      mat0 = now_ms();
      if (compute_checksum)
      {
#pragma omp parallel for schedule(static)
         for (int lq_i = 0; lq_i < static_cast<int>(qn); ++lq_i)
         {
            Id lq = static_cast<Id>(lq_i);
            Id count = std::min<Id>(h_output_counts[lq], static_cast<Id>(compact_output_cap));
            const Id *ids = h_output_ids + h_output_offsets[lq];
            std::vector<Id> sorted_ids(ids, ids + count);
            std::sort(sorted_ids.begin(), sorted_ids.end());
            uint64_t h = 1469598103934665603ull;
            for (Id gid : sorted_ids)
               h = (h ^ gid) * 1099511628211ull;
            checksums[qb + lq] = h;
         }
      }
      r.prune_ms += now_ms() - mat0;
   }

   r.total_ms = now_ms() - t0;
   CUDA_CHECK(cudaFree(d_candidate_bits));
   CUDA_CHECK(cudaFree(d_frontier_bits));
   CUDA_CHECK(cudaFree(d_selected_frontier_bits));
   CUDA_CHECK(cudaFree(d_covered_bits));
   CUDA_CHECK(cudaFree(d_out_bits));
   CUDA_CHECK(cudaFree(d_frontier_counts));
   CUDA_CHECK(cudaFree(d_frontier_ids));
   CUDA_CHECK(cudaFree(d_output_counts));
   CUDA_CHECK(cudaFree(d_output_offsets));
   CUDA_CHECK(cudaFree(d_output_ids));
   CUDA_CHECK(cudaFreeHost(h_output_counts));
   CUDA_CHECK(cudaFreeHost(h_output_offsets));
   CUDA_CHECK(cudaFreeHost(h_output_ids));
   CUDA_CHECK(cudaFreeHost(h_frontier_counts));
   CUDA_CHECK(cudaEventDestroy(ev0));
   CUDA_CHECK(cudaEventDestroy(ev1));
   r.avg_frontier_groups = static_cast<double>(frontier_count_sum) / queries.size();
   r.avg_candidates = std::accumulate(output_counts.begin(), output_counts.end(), 0.0) / queries.size();
   r.avg_min_groups = r.avg_candidates;
   r.avg_output_groups = r.avg_candidates;
   if (compute_checksum)
      r.checksum = std::accumulate(checksums.begin(), checksums.end(), uint64_t{0}, std::bit_xor<uint64_t>());
   return r;
}

BenchResult run_cpu_cover_frontier_reference(const GroupTable &groups, const LabelTable &dense_queries,
                                             const LabelBitsets &bitsets, const DescendantBitsets &descendants,
                                             int frontier_delta, int frontier_cover_cap, int threads)
{
   BenchResult r;
   r.provider = "cpu_cover_frontier_ref_d" + std::to_string(frontier_delta) +
                "_cap" + std::to_string(frontier_cover_cap);
   const Id words = bitsets.words_per_query;
   std::vector<uint64_t> counts(dense_queries.size(), 0);
   std::vector<uint64_t> checksums(dense_queries.size(), 0);
   std::vector<uint64_t> frontier_counts(dense_queries.size(), 0);
   std::vector<std::vector<Id>> output_group_ids(dense_queries.size());
   omp_set_num_threads(threads);

   double t0 = now_ms();
#pragma omp parallel for schedule(dynamic, 8)
   for (int qi_i = 0; qi_i < static_cast<int>(dense_queries.size()); ++qi_i)
   {
      Id qi = static_cast<Id>(qi_i);
      std::vector<uint64_t> candidates(words, ~uint64_t{0});
      Id q_begin = dense_queries.offsets[qi];
      Id q_end = dense_queries.offsets[qi + 1];
      for (Id off = q_begin; off < q_end; ++off)
      {
         Label dense = dense_queries.labels[off];
         if (dense == kInvalidDenseLabel)
         {
            std::fill(candidates.begin(), candidates.end(), 0);
            break;
         }
         const uint64_t *label_bits = bitsets.label_group_bits.data() + static_cast<size_t>(dense) * words;
         for (Id word = 0; word < words; ++word)
            candidates[word] &= label_bits[word];
      }

      std::vector<uint64_t> frontier(words, 0);
      Id q_len = q_end - q_begin;
      Id end_len = std::min<Id>(bitsets.max_group_label_size, q_len + frontier_delta);
      for (Id len = q_len; len <= end_len; ++len)
      {
         const uint64_t *size_bits = bitsets.size_group_bits.data() + static_cast<size_t>(len) * words;
         for (Id word = 0; word < words; ++word)
            frontier[word] |= candidates[word] & size_bits[word];
      }

      std::vector<uint64_t> covered(words, 0);
      std::vector<uint64_t> selected_frontier(words, 0);
      Id frontier_count = 0;
      for (Id fword = 0; fword < words; ++fword)
      {
         uint64_t bits = frontier[fword];
         while (bits)
         {
            Id bit = static_cast<Id>(__builtin_ctzll(bits));
            Id gid = (fword << 6) + bit;
            if (gid > 0 && gid < groups.table.size())
            {
               if (frontier_count < static_cast<Id>(frontier_cover_cap))
               {
                  ++frontier_count;
                  selected_frontier[fword] |= uint64_t{1} << bit;
                  const uint64_t *desc = descendants.group_desc_bits.data() + static_cast<size_t>(gid) * words;
                  for (Id word = 0; word < words; ++word)
                     covered[word] |= desc[word];
               }
            }
            bits &= bits - 1;
         }
      }
      frontier_counts[qi] = frontier_count;

      uint64_t h = 1469598103934665603ull;
      uint64_t count = 0;
      auto &dst = output_group_ids[qi];
      dst.clear();
      for (Id word = 0; word < words; ++word)
      {
         uint64_t out = selected_frontier[word] | (candidates[word] & ~covered[word]);
         while (out)
         {
            Id bit = static_cast<Id>(__builtin_ctzll(out));
            Id gid = (word << 6) + bit;
            if (gid > 0 && gid < groups.table.size())
            {
               ++count;
               dst.push_back(gid);
               h = (h ^ gid) * 1099511628211ull;
            }
            out &= out - 1;
         }
      }
      counts[qi] = count;
      checksums[qi] = h;
   }
   r.total_ms = now_ms() - t0;
   r.kernel_ms = r.total_ms;
   r.avg_frontier_groups = std::accumulate(frontier_counts.begin(), frontier_counts.end(), 0.0) / dense_queries.size();
   r.avg_candidates = std::accumulate(counts.begin(), counts.end(), 0.0) / dense_queries.size();
   r.avg_min_groups = r.avg_candidates;
   r.avg_output_groups = r.avg_min_groups;
   r.checksum = std::accumulate(checksums.begin(), checksums.end(), uint64_t{0}, std::bit_xor<uint64_t>());
   return r;
}

BenchResult best_of(const std::vector<BenchResult> &runs)
{
   return *std::min_element(runs.begin(), runs.end(), [](const BenchResult &a, const BenchResult &b) {
      return a.total_ms < b.total_ms;
   });
}

void print_result(const BenchResult &r, Id nq, Id ngroups)
{
   std::cout << "provider=" << r.provider
             << " nq=" << nq
             << " ngroups=" << ngroups
             << " total_ms=" << r.total_ms
             << " kernel_ms=" << r.kernel_ms
             << " d2h_ms=" << r.d2h_ms
             << " prune_ms=" << r.prune_ms
             << " avg_frontier_groups=" << r.avg_frontier_groups
             << " avg_candidates=" << r.avg_candidates
             << " avg_min_groups=" << r.avg_min_groups
             << " avg_output_groups=" << r.avg_output_groups
             << " output_overflow_queries=" << r.output_overflow_queries
             << " output_truncated_groups=" << r.output_truncated_groups
             << " checksum=" << r.checksum << "\n";
}

void write_csv(const std::string &path, const std::vector<BenchResult> &results, Id nq, Id ngroups)
{
   if (path.empty())
      return;
   std::ofstream out(path);
   if (!out)
      throw std::runtime_error("failed to open output csv: " + path);
   out << "provider,nq,ngroups,total_ms,kernel_ms,d2h_ms,prune_ms,avg_frontier_groups,avg_candidates,avg_min_groups,avg_output_groups,output_overflow_queries,output_truncated_groups,checksum\n";
   for (const auto &r : results)
   {
      out << r.provider << "," << nq << "," << ngroups << ","
          << r.total_ms << "," << r.kernel_ms << "," << r.d2h_ms << ","
          << r.prune_ms << "," << r.avg_frontier_groups << ","
          << r.avg_candidates << "," << r.avg_min_groups << ","
          << r.avg_output_groups << ","
          << r.output_overflow_queries << ","
          << r.output_truncated_groups << ","
          << r.checksum << "\n";
   }
}

} // namespace

int main(int argc, char **argv)
{
   try
   {
      Options opt = parse_args(argc, argv);
      omp_set_num_threads(opt.threads);
      CUDA_CHECK(cudaSetDevice(opt.gpu));

      auto base_rows = read_label_sets(opt.base_label_file, opt.max_base);
      auto query_rows = read_label_sets(opt.query_label_file, opt.max_query);
      GroupTable groups = build_groups(base_rows);
      LabelTable queries = flatten_rows(query_rows);
      std::cout << "loaded base_rows=" << base_rows.size()
                << " query_rows=" << query_rows.size()
                << " unique_groups=" << groups.table.size() - 1
                << " base_label_values=" << groups.table.labels.size()
                << " query_label_values=" << queries.labels.size() << "\n";

      std::vector<BenchResult> final_results;
      BenchResult cpu_ref;
      bool no_exact_checksum = opt.provider == "gpu_cover_frontier" || opt.provider == "gpu_cover_frontier_compact";
      bool need_cpu_exact = opt.provider == "all" || opt.provider == "cpu_exact" || (opt.check && !no_exact_checksum && opt.provider != "cpu_scan");
      if (opt.provider == "all" || opt.provider == "cpu_scan")
      {
         std::vector<BenchResult> runs;
         for (int i = 0; i < opt.warmup + opt.repeats; ++i)
         {
            auto r = run_cpu_scan(groups, queries, opt.threads);
            if (i >= opt.warmup)
               runs.push_back(r);
         }
         auto cpu_scan = best_of(runs);
         final_results.push_back(cpu_scan);
         print_result(cpu_scan, queries.size(), groups.table.size() - 1);
      }
      if (need_cpu_exact)
      {
         std::vector<BenchResult> runs;
         for (int i = 0; i < opt.warmup + opt.repeats; ++i)
         {
            auto r = run_cpu_exact(groups, queries, opt.threads);
            if (i >= opt.warmup)
               runs.push_back(r);
         }
         cpu_ref = best_of(runs);
         if (opt.provider == "all" || opt.provider == "cpu_exact")
         {
            final_results.push_back(cpu_ref);
            print_result(cpu_ref, queries.size(), groups.table.size() - 1);
         }
      }

      if (opt.provider == "all" || opt.provider == "gpu_scan")
      {
         DeviceTables dev(groups, queries);
         std::vector<BenchResult> runs;
         for (int i = 0; i < opt.warmup + opt.repeats; ++i)
         {
            auto r = run_gpu_scan_once(groups, queries, dev, opt.query_batch, opt.group_chunk);
            if (i >= opt.warmup)
               runs.push_back(r);
         }
         auto gpu = best_of(runs);
         if (opt.check && gpu.checksum != cpu_ref.checksum)
         {
            std::cerr << "[ERROR] gpu_scan checksum mismatch: gpu=" << gpu.checksum
                      << " cpu=" << cpu_ref.checksum << "\n";
            return 2;
         }
         final_results.push_back(gpu);
         print_result(gpu, queries.size(), groups.table.size() - 1);
      }

      if (opt.provider == "all" || opt.provider == "gpu_bitset")
      {
         LabelBitsets bitsets = build_label_bitsets(groups);
         LabelTable dense_queries = remap_queries_to_dense(queries, bitsets.label_to_dense);
         DeviceBitsetTables dev(bitsets, dense_queries);
         std::vector<BenchResult> runs;
         for (int i = 0; i < opt.warmup + opt.repeats; ++i)
         {
            auto r = run_gpu_bitset_once(groups, queries, bitsets, dev, opt.query_batch);
            if (i >= opt.warmup)
               runs.push_back(r);
         }
         auto gpu = best_of(runs);
         if (opt.check && gpu.checksum != cpu_ref.checksum)
         {
            std::cerr << "[ERROR] gpu_bitset checksum mismatch: gpu=" << gpu.checksum
                      << " cpu=" << cpu_ref.checksum << "\n";
            return 2;
         }
         final_results.push_back(gpu);
         print_result(gpu, queries.size(), groups.table.size() - 1);
      }

      if (opt.provider == "cover_frontier_sweep")
      {
         {
            std::vector<BenchResult> runs;
            for (int i = 0; i < opt.warmup + opt.repeats; ++i)
            {
               auto r = run_cpu_scan(groups, queries, opt.threads);
               if (i >= opt.warmup)
                  runs.push_back(r);
            }
            auto cpu_scan = best_of(runs);
            final_results.push_back(cpu_scan);
            print_result(cpu_scan, queries.size(), groups.table.size() - 1);
         }

         {
            std::vector<BenchResult> runs;
            for (int i = 0; i < opt.warmup + opt.repeats; ++i)
            {
               auto r = run_cpu_exact(groups, queries, opt.threads);
               if (i >= opt.warmup)
                  runs.push_back(r);
            }
            auto cpu_exact = best_of(runs);
            final_results.push_back(cpu_exact);
            print_result(cpu_exact, queries.size(), groups.table.size() - 1);
         }

         LabelBitsets bitsets = build_label_bitsets(groups);
         LabelTable dense_queries = remap_queries_to_dense(queries, bitsets.label_to_dense);
         double desc0 = now_ms();
         DescendantBitsets descendants = build_descendant_bitsets_from_labels(groups, bitsets);
         double desc_build_ms = now_ms() - desc0;
         std::cout << "descendant_bitsets built groups=" << descendants.num_groups - 1
                   << " words=" << descendants.words_per_query
                   << " bytes=" << descendants.group_desc_bits.size() * sizeof(uint64_t)
                   << " build_ms=" << desc_build_ms << "\n";
         DeviceBitsetTables dev(bitsets, dense_queries);
         DeviceDescendantBitsets dev_desc(descendants);

         for (int delta : opt.frontier_deltas)
         {
            for (int cap : opt.frontier_cover_caps)
            {
               std::vector<BenchResult> cpu_cover_runs;
               for (int i = 0; i < opt.warmup + opt.repeats; ++i)
               {
                  auto r = run_cpu_cover_frontier_reference(groups, dense_queries, bitsets, descendants,
                                                            delta, cap, opt.threads);
                  if (i >= opt.warmup)
                     cpu_cover_runs.push_back(r);
               }
               auto cpu_cover = best_of(cpu_cover_runs);
               final_results.push_back(cpu_cover);
               print_result(cpu_cover, queries.size(), groups.table.size() - 1);

               std::vector<BenchResult> gpu_cover_runs;
               for (int i = 0; i < opt.warmup + opt.repeats; ++i)
               {
                  auto r = run_gpu_cover_frontier_once(groups, queries, bitsets, dev, dev_desc,
                                                       opt.query_batch, delta, cap);
                  if (i >= opt.warmup)
                     gpu_cover_runs.push_back(r);
               }
               auto gpu_cover = best_of(gpu_cover_runs);
               if (opt.check && gpu_cover.checksum != cpu_cover.checksum)
               {
                  std::cerr << "[ERROR] cover_frontier_sweep checksum mismatch delta=" << delta
                            << " cap=" << cap
                            << " gpu=" << gpu_cover.checksum
                            << " cpu=" << cpu_cover.checksum << "\n";
                  return 2;
               }
               final_results.push_back(gpu_cover);
               print_result(gpu_cover, queries.size(), groups.table.size() - 1);

               std::vector<BenchResult> gpu_compact_runs;
               for (int i = 0; i < opt.warmup + opt.repeats; ++i)
               {
                  auto r = run_gpu_cover_frontier_compact_once(groups, queries, bitsets, dev, dev_desc,
                                                               opt.query_batch, delta, cap,
                                                               opt.compact_output_cap, opt.check);
                  if (i >= opt.warmup)
                     gpu_compact_runs.push_back(r);
               }
               auto gpu_compact = best_of(gpu_compact_runs);
               final_results.push_back(gpu_compact);
               print_result(gpu_compact, queries.size(), groups.table.size() - 1);
            }
         }
      }

      if (opt.provider == "all" || opt.provider == "gpu_cover_frontier" ||
          opt.provider == "gpu_cover_frontier_compact" || opt.provider == "cpu_cover_frontier")
      {
         LabelBitsets bitsets = build_label_bitsets(groups);
         LabelTable dense_queries = remap_queries_to_dense(queries, bitsets.label_to_dense);
         double desc0 = now_ms();
         DescendantBitsets descendants = build_descendant_bitsets_from_labels(groups, bitsets);
         double desc_build_ms = now_ms() - desc0;
         std::cout << "descendant_bitsets built groups=" << descendants.num_groups - 1
                   << " words=" << descendants.words_per_query
                   << " bytes=" << descendants.group_desc_bits.size() * sizeof(uint64_t)
                   << " build_ms=" << desc_build_ms << "\n";
         BenchResult cpu_cover_ref;
         bool need_cpu_cover_ref = opt.provider == "all" || opt.provider == "cpu_cover_frontier" ||
                                   (opt.check && opt.provider == "gpu_cover_frontier");
         if (need_cpu_cover_ref)
         {
            std::vector<BenchResult> cpu_cover_runs;
            for (int i = 0; i < opt.warmup + opt.repeats; ++i)
            {
               auto r = run_cpu_cover_frontier_reference(groups, dense_queries, bitsets, descendants,
                                                         opt.frontier_delta, opt.frontier_cover_cap, opt.threads);
               if (i >= opt.warmup)
                  cpu_cover_runs.push_back(r);
            }
            cpu_cover_ref = best_of(cpu_cover_runs);
            if (opt.provider == "all" || opt.provider == "cpu_cover_frontier")
            {
               final_results.push_back(cpu_cover_ref);
               print_result(cpu_cover_ref, queries.size(), groups.table.size() - 1);
            }
         }
         if (opt.provider == "all" || opt.provider == "gpu_cover_frontier" || opt.provider == "gpu_cover_frontier_compact")
         {
            DeviceBitsetTables dev(bitsets, dense_queries);
            DeviceDescendantBitsets dev_desc(descendants);
            if (opt.provider == "all" || opt.provider == "gpu_cover_frontier")
            {
               std::vector<BenchResult> runs;
               for (int i = 0; i < opt.warmup + opt.repeats; ++i)
               {
                  auto r = run_gpu_cover_frontier_once(groups, queries, bitsets, dev, dev_desc,
                                                       opt.query_batch, opt.frontier_delta,
                                                       opt.frontier_cover_cap);
                  if (i >= opt.warmup)
                     runs.push_back(r);
               }
               auto gpu = best_of(runs);
               if (opt.check && gpu.checksum != cpu_cover_ref.checksum)
               {
                  std::cerr << "[ERROR] gpu_cover_frontier checksum mismatch: gpu=" << gpu.checksum
                            << " cpu_ref=" << cpu_cover_ref.checksum << "\n";
                  return 2;
               }
               final_results.push_back(gpu);
               print_result(gpu, queries.size(), groups.table.size() - 1);
            }
            if (opt.provider == "all" || opt.provider == "gpu_cover_frontier_compact")
            {
               std::vector<BenchResult> runs;
               for (int i = 0; i < opt.warmup + opt.repeats; ++i)
               {
               auto r = run_gpu_cover_frontier_compact_once(groups, queries, bitsets, dev, dev_desc,
                                                            opt.query_batch, opt.frontier_delta,
                                                            opt.frontier_cover_cap,
                                                            opt.compact_output_cap, opt.check);
                  if (i >= opt.warmup)
                     runs.push_back(r);
               }
               auto gpu = best_of(runs);
               final_results.push_back(gpu);
               print_result(gpu, queries.size(), groups.table.size() - 1);
            }
         }
      }

      write_csv(opt.output_csv, final_results, queries.size(), groups.table.size() - 1);
      return 0;
   }
   catch (const std::exception &e)
   {
      std::cerr << "[ERROR] " << e.what() << "\n";
      return 1;
   }
}
