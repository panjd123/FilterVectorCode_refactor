#include "include/ung_special_edge_io.h"

#include <algorithm>
#include <charconv>
#include <cstdio>
#include <fstream>
#include <limits>
#include <system_error>
#include <vector>

namespace ANNS
{
namespace
{

constexpr char kCsrMagic[8] = {'S', 'B', 'C', 'S', 'R', '0', '2', '\0'};
constexpr uint32_t kCsrVersion = 2;
constexpr uint32_t kRegularTargetsOnly = 1;
constexpr uint32_t kSpecialPayload = 2;

struct CsrHeader
{
   char magic[8];
   uint32_t version = kCsrVersion;
   uint32_t payload_kind = 0;
   uint64_t num_sources = 0;
   uint64_t edge_count = 0;
};

struct CompactSpecialEdge
{
   uint32_t target = 0;
   uint32_t block_and_kind = 0;
};

static_assert(sizeof(CsrHeader) == 32, "CSR edge header must stay 32 bytes");
static_assert(sizeof(CompactSpecialEdge) == 8, "compact special edge must stay 8 bytes");
static_assert(sizeof(CompactSpecialEdge) == sizeof(PackedSpecialEdge),
              "disk and runtime packed special edges must match");
static_assert(sizeof(IdxType) == sizeof(uint32_t), "CSR target IDs require 32-bit IdxType");

bool read_csr_header(std::ifstream &in, CsrHeader &header)
{
   in.seekg(0);
   in.read(reinterpret_cast<char *>(&header), sizeof(header));
   return static_cast<bool>(in) &&
          std::equal(std::begin(header.magic), std::end(header.magic), std::begin(kCsrMagic));
}

uint64_t csr_expected_size(const CsrHeader &header, size_t edge_bytes)
{
   if (header.num_sources == std::numeric_limits<uint64_t>::max())
      return std::numeric_limits<uint64_t>::max();
   const uint64_t offsets = header.num_sources + 1;
   if (offsets > (std::numeric_limits<uint64_t>::max() - sizeof(CsrHeader)) / sizeof(uint64_t))
      return std::numeric_limits<uint64_t>::max();
   const uint64_t prefix = sizeof(CsrHeader) + offsets * sizeof(uint64_t);
   if (header.edge_count > (std::numeric_limits<uint64_t>::max() - prefix) / edge_bytes)
      return std::numeric_limits<uint64_t>::max();
   return prefix + header.edge_count * edge_bytes;
}

template <typename EdgeWriter>
bool write_csr_file(const std::string &path,
                    uint32_t payload_kind,
                    uint64_t num_sources,
                    const std::vector<uint64_t> &offsets,
                    uint64_t count,
                    EdgeWriter write_edges,
                    std::string &error)
{
   const std::string temporary_path = path + ".tmp";
   std::ofstream out(temporary_path, std::ios::binary | std::ios::trunc);
   if (!out)
   {
      error = "cannot open temporary CSR output";
      return false;
   }
   CsrHeader header;
   std::copy(std::begin(kCsrMagic), std::end(kCsrMagic), std::begin(header.magic));
   header.payload_kind = payload_kind;
   header.num_sources = num_sources;
   header.edge_count = count;
   out.write(reinterpret_cast<const char *>(&header), sizeof(header));
   out.write(reinterpret_cast<const char *>(offsets.data()),
             static_cast<std::streamsize>(offsets.size() * sizeof(uint64_t)));
   write_edges(out);
   out.flush();
   out.close();
   const auto validation = validate_special_edge_binary_file(temporary_path);
   if (!validation.valid || validation.count != count)
   {
      error = validation.valid ? "temporary CSR count mismatch" : validation.error;
      std::remove(temporary_path.c_str());
      return false;
   }
   if (std::rename(temporary_path.c_str(), path.c_str()) != 0)
   {
      error = "cannot rename validated temporary CSR output";
      std::remove(temporary_path.c_str());
      return false;
   }
   error.clear();
   return true;
}

bool parse_uint_field(const char *&cursor, const char *end, uint32_t &value)
{
   const auto parsed = std::from_chars(cursor, end, value);
   if (parsed.ec != std::errc{} || parsed.ptr == end || *parsed.ptr != ',')
      return false;
   cursor = parsed.ptr + 1;
   return true;
}

} // namespace

bool parse_special_edge_csv_record(const std::string &raw_line,
                                   SpecialEdgeBinaryRecord &record,
                                   std::string &error)
{
   std::string line = raw_line;
   if (!line.empty() && line.back() == '\r')
      line.pop_back();
   if (line.empty())
   {
      error = "CSV edge row is empty";
      return false;
   }

   const char *cursor = line.data();
   const char *end = cursor + line.size();
   SpecialEdgeBinaryRecord parsed_record;
   if (!parse_uint_field(cursor, end, parsed_record.source) ||
       !parse_uint_field(cursor, end, parsed_record.target) ||
       !parse_uint_field(cursor, end, parsed_record.block))
   {
      error = "CSV edge row must contain three unsigned integer fields";
      return false;
   }
   const std::string kind(cursor, end);
   if (kind == "inter")
      parsed_record.kind = 1;
   else if (kind == "intra")
      parsed_record.kind = 0;
   else
   {
      error = "CSV edge kind must be intra or inter";
      return false;
   }

   record = parsed_record;
   error.clear();
   return true;
}

SpecialEdgeBinaryValidation validate_special_edge_binary_file(const std::string &path)
{
   SpecialEdgeBinaryValidation validation;
   std::ifstream in(path, std::ios::binary | std::ios::ate);
   if (!in)
   {
      validation.error = "cannot open binary file";
      return validation;
   }

   const std::streamoff size = in.tellg();
   CsrHeader csr_header;
   if (size >= static_cast<std::streamoff>(sizeof(CsrHeader)) && read_csr_header(in, csr_header))
   {
      if (csr_header.version != kCsrVersion ||
          (csr_header.payload_kind != kRegularTargetsOnly && csr_header.payload_kind != kSpecialPayload))
      {
         validation.error = "unsupported CSR edge format";
         return validation;
      }
      const size_t edge_bytes = csr_header.payload_kind == kRegularTargetsOnly
                                    ? sizeof(uint32_t)
                                    : sizeof(CompactSpecialEdge);
      if (static_cast<uint64_t>(size) != csr_expected_size(csr_header, edge_bytes))
      {
         validation.error = "CSR file size does not match its header";
         return validation;
      }
      in.seekg(sizeof(CsrHeader));
      std::vector<uint64_t> offsets(static_cast<size_t>(csr_header.num_sources) + 1);
      in.read(reinterpret_cast<char *>(offsets.data()),
              static_cast<std::streamsize>(offsets.size() * sizeof(uint64_t)));
      if (!in || offsets.empty() || offsets.front() != 0 || offsets.back() != csr_header.edge_count ||
          !std::is_sorted(offsets.begin(), offsets.end()))
      {
         validation.error = "invalid CSR offsets";
         return validation;
      }
      validation.valid = true;
      validation.count = csr_header.edge_count;
      validation.num_sources = csr_header.num_sources;
      validation.format_version = kCsrVersion;
      validation.regular_targets_only = csr_header.payload_kind == kRegularTargetsOnly;
      return validation;
   }
   if (size < static_cast<std::streamoff>(sizeof(uint64_t)))
   {
      validation.error = "binary file is shorter than its count header";
      return validation;
   }
   in.seekg(0);
   uint64_t count = 0;
   in.read(reinterpret_cast<char *>(&count), sizeof(count));
   if (!in)
   {
      validation.error = "cannot read binary count header";
      return validation;
   }
   if (count > (std::numeric_limits<uint64_t>::max() - sizeof(uint64_t)) /
                   sizeof(SpecialEdgeBinaryRecord))
   {
      validation.error = "binary edge count overflows expected file size";
      return validation;
   }
   const uint64_t expected = sizeof(uint64_t) + count * sizeof(SpecialEdgeBinaryRecord);
   if (static_cast<uint64_t>(size) != expected)
   {
      validation.error = "binary file size does not match declared edge count";
      return validation;
   }

   validation.valid = true;
   validation.count = count;
   validation.format_version = 1;
   return validation;
}

bool for_each_special_edge_binary_record(const std::string &path,
                                         const SpecialEdgeRecordConsumer &consumer,
                                         uint64_t &count,
                                         std::string &error)
{
   const SpecialEdgeBinaryValidation validation = validate_special_edge_binary_file(path);
   if (!validation.valid)
   {
      error = validation.error;
      return false;
   }

   std::ifstream in(path, std::ios::binary);
   if (validation.format_version == kCsrVersion)
   {
      CsrHeader header;
      if (!read_csr_header(in, header))
      {
         error = "cannot read validated CSR header";
         return false;
      }
      std::vector<uint64_t> offsets(static_cast<size_t>(header.num_sources) + 1);
      in.read(reinterpret_cast<char *>(offsets.data()),
              static_cast<std::streamsize>(offsets.size() * sizeof(uint64_t)));
      for (uint64_t source = 0; source < header.num_sources; ++source)
      {
         for (uint64_t edge_idx = offsets[source]; edge_idx < offsets[source + 1]; ++edge_idx)
         {
            SpecialEdgeBinaryRecord record;
            record.source = static_cast<uint32_t>(source);
            if (validation.regular_targets_only)
            {
               in.read(reinterpret_cast<char *>(&record.target), sizeof(record.target));
            }
            else
            {
               CompactSpecialEdge compact;
               in.read(reinterpret_cast<char *>(&compact), sizeof(compact));
               record.target = compact.target;
               record.kind = static_cast<uint8_t>(compact.block_and_kind & 1U);
               record.block = compact.block_and_kind >> 1U;
            }
            if (!in)
            {
               error = "CSR payload read failed after validation";
               return false;
            }
            consumer(record);
         }
      }
      count = validation.count;
      error.clear();
      return true;
   }
   in.seekg(sizeof(uint64_t));
   for (uint64_t i = 0; i < validation.count; ++i)
   {
      SpecialEdgeBinaryRecord record;
      in.read(reinterpret_cast<char *>(&record), sizeof(record));
      if (!in)
      {
         error = "binary record read failed after validation";
         return false;
      }
      consumer(record);
   }
   count = validation.count;
   error.clear();
   return true;
}

bool write_regular_edge_csr_file(
    const std::string &path,
    const std::vector<std::vector<IdxType>> &edges_by_source,
    uint64_t &count,
    std::string &error)
{
   std::vector<uint64_t> offsets(edges_by_source.size() + 1, 0);
   for (size_t source = 0; source < edges_by_source.size(); ++source)
      offsets[source + 1] = offsets[source] + edges_by_source[source].size();
   count = offsets.back();
   return write_csr_file(
       path, kRegularTargetsOnly, edges_by_source.size(), offsets, count,
       [&](std::ofstream &out) {
          for (const auto &edges : edges_by_source)
             for (IdxType target : edges)
             {
                const uint32_t value = static_cast<uint32_t>(target);
                out.write(reinterpret_cast<const char *>(&value), sizeof(value));
             }
       },
       error);
}

bool write_special_edge_csr_file(
    const std::string &path,
    const std::vector<std::vector<SpecialEdge>> &edges_by_source,
    const std::function<bool(IdxType, const SpecialEdge &)> &include,
    uint64_t &count,
    std::string &error)
{
   std::vector<uint64_t> offsets(edges_by_source.size() + 1, 0);
   for (IdxType source = 0; source < edges_by_source.size(); ++source)
   {
      uint64_t included = 0;
      for (const auto &edge : edges_by_source[source])
         included += include(source, edge) ? 1 : 0;
      offsets[source + 1] = offsets[source] + included;
   }
   count = offsets.back();
   return write_csr_file(
       path, kSpecialPayload, edges_by_source.size(), offsets, count,
       [&](std::ofstream &out) {
          for (IdxType source = 0; source < edges_by_source.size(); ++source)
             for (const auto &edge : edges_by_source[source])
             {
                if (!include(source, edge))
                   continue;
                CompactSpecialEdge compact;
                compact.target = static_cast<uint32_t>(edge.target_point_id);
                compact.block_and_kind =
                    (static_cast<uint32_t>(edge.special_block_id) << 1U) |
                    (edge.kind == SpecialEdgeKind::InterBlock ? 1U : 0U);
                out.write(reinterpret_cast<const char *>(&compact), sizeof(compact));
             }
       },
       error);
}

bool read_regular_edge_csr_file(
    const std::string &path,
    RegularEdgeCsr &csr,
    std::string &error)
{
   const auto validation = validate_special_edge_binary_file(path);
   if (!validation.valid || validation.format_version != kCsrVersion ||
       !validation.regular_targets_only)
   {
      error = validation.valid ? "file is not a regular-edge CSR" : validation.error;
      return false;
   }
   std::ifstream in(path, std::ios::binary);
   CsrHeader header;
   if (!read_csr_header(in, header))
   {
      error = "cannot read regular-edge CSR header";
      return false;
   }
   csr.offsets.resize(static_cast<size_t>(header.num_sources) + 1);
   csr.targets.resize(static_cast<size_t>(header.edge_count));
   in.read(reinterpret_cast<char *>(csr.offsets.data()),
           static_cast<std::streamsize>(csr.offsets.size() * sizeof(uint64_t)));
   in.read(reinterpret_cast<char *>(csr.targets.data()),
           static_cast<std::streamsize>(csr.targets.size() * sizeof(IdxType)));
   if (!in)
   {
      csr.clear();
      error = "cannot read regular-edge CSR payload";
      return false;
   }
   error.clear();
   return true;
}

bool read_special_edge_csr_file(
    const std::string &path,
    SpecialEdgeCsr &csr,
    std::string &error)
{
   const auto validation = validate_special_edge_binary_file(path);
   if (!validation.valid || validation.format_version != kCsrVersion ||
       validation.regular_targets_only)
   {
      error = validation.valid ? "file is not a special-edge CSR" : validation.error;
      return false;
   }
   std::ifstream in(path, std::ios::binary);
   CsrHeader header;
   if (!read_csr_header(in, header))
   {
      error = "cannot read special-edge CSR header";
      return false;
   }
   csr.offsets.resize(static_cast<size_t>(header.num_sources) + 1);
   csr.edges.resize(static_cast<size_t>(header.edge_count));
   in.read(reinterpret_cast<char *>(csr.offsets.data()),
           static_cast<std::streamsize>(csr.offsets.size() * sizeof(uint64_t)));
   in.read(reinterpret_cast<char *>(csr.edges.data()),
           static_cast<std::streamsize>(csr.edges.size() * sizeof(PackedSpecialEdge)));
   if (!in)
   {
      csr.clear();
      error = "cannot read special-edge CSR payload";
      return false;
   }
   error.clear();
   return true;
}

bool convert_special_edge_csv_to_binary(const std::string &csv_path,
                                        const std::string &binary_path,
                                        uint64_t &count,
                                        std::string &error)
{
   std::ifstream in(csv_path);
   if (!in)
   {
      error = "cannot open CSV input";
      return false;
   }

   const std::string temporary_path = binary_path + ".tmp";
   std::ofstream out(temporary_path, std::ios::binary | std::ios::trunc);
   if (!out)
   {
      error = "cannot open temporary binary output";
      return false;
   }

   uint64_t written = 0;
   out.write(reinterpret_cast<const char *>(&written), sizeof(written));
   std::string line;
   std::getline(in, line);
   uint64_t line_number = 1;
   while (std::getline(in, line))
   {
      ++line_number;
      if (line.empty())
         continue;

      SpecialEdgeBinaryRecord record;
      std::string parse_error;
      if (!parse_special_edge_csv_record(line, record, parse_error))
      {
         error = parse_error + " at line " + std::to_string(line_number);
         out.close();
         std::remove(temporary_path.c_str());
         return false;
      }
      out.write(reinterpret_cast<const char *>(&record), sizeof(record));
      if (!out)
      {
         error = "failed while writing temporary binary output";
         out.close();
         std::remove(temporary_path.c_str());
         return false;
      }
      ++written;
   }

   out.seekp(0);
   out.write(reinterpret_cast<const char *>(&written), sizeof(written));
   out.flush();
   out.close();
   const SpecialEdgeBinaryValidation validation = validate_special_edge_binary_file(temporary_path);
   if (!validation.valid || validation.count != written)
   {
      error = validation.valid ? "temporary binary count mismatch" : validation.error;
      std::remove(temporary_path.c_str());
      return false;
   }
   if (std::rename(temporary_path.c_str(), binary_path.c_str()) != 0)
   {
      error = "cannot rename validated temporary binary output";
      std::remove(temporary_path.c_str());
      return false;
   }

   count = written;
   error.clear();
   return true;
}

} // namespace ANNS
