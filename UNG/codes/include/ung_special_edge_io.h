#ifndef ANNS_UNG_SPECIAL_EDGE_IO_H
#define ANNS_UNG_SPECIAL_EDGE_IO_H

#include "config.h"
#include "ung_special_blocks.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace ANNS
{

struct SpecialEdgeBinaryRecord
{
   uint32_t source = 0;
   uint32_t target = 0;
   uint32_t block = 0;
   uint8_t kind = 0;
   uint8_t pad[3] = {0, 0, 0};
};

static_assert(sizeof(SpecialEdgeBinaryRecord) == 16, "special-edge binary record must stay 16 bytes");

struct SpecialEdgeBinaryValidation
{
   bool valid = false;
   uint64_t count = 0;
   uint64_t num_sources = 0;
   uint32_t format_version = 0;
   bool regular_targets_only = false;
   std::string error;
};

struct RegularEdgeCsr
{
   std::vector<uint64_t> offsets;
   std::vector<IdxType> targets;

   void clear()
   {
      offsets.clear();
      targets.clear();
   }
   bool empty() const { return offsets.empty(); }
   size_t num_sources() const { return offsets.empty() ? 0 : offsets.size() - 1; }
};

struct PackedSpecialEdge
{
   uint32_t target = 0;
   uint32_t block_and_kind = 0;

   SpecialEdge unpack() const
   {
      return {static_cast<IdxType>(target),
              static_cast<IdxType>(block_and_kind >> 1U),
              (block_and_kind & 1U) != 0
                  ? SpecialEdgeKind::InterBlock
                  : SpecialEdgeKind::IntraBlock};
   }
};

static_assert(sizeof(PackedSpecialEdge) == 8, "packed special edge must stay 8 bytes");

struct SpecialEdgeCsr
{
   std::vector<uint64_t> offsets;
   std::vector<PackedSpecialEdge> edges;

   void clear()
   {
      offsets.clear();
      edges.clear();
   }
   bool empty() const { return offsets.empty(); }
   size_t num_sources() const { return offsets.empty() ? 0 : offsets.size() - 1; }
};

using SpecialEdgeRecordConsumer = std::function<void(const SpecialEdgeBinaryRecord &)>;

// Parse one non-empty legacy CSV data row. The parser is deliberately shared
// by conversion and runtime fallback loading so both paths reject malformed
// rows instead of silently accepting different subsets of an edge file.
bool parse_special_edge_csv_record(const std::string &line,
                                   SpecialEdgeBinaryRecord &record,
                                   std::string &error);
SpecialEdgeBinaryValidation validate_special_edge_binary_file(const std::string &path);
bool for_each_special_edge_binary_record(const std::string &path,
                                         const SpecialEdgeRecordConsumer &consumer,
                                         uint64_t &count,
                                         std::string &error);
bool write_regular_edge_csr_file(
    const std::string &path,
    const std::vector<std::vector<IdxType>> &edges_by_source,
    uint64_t &count,
    std::string &error);
bool write_special_edge_csr_file(
    const std::string &path,
    const std::vector<std::vector<SpecialEdge>> &edges_by_source,
    const std::function<bool(IdxType, const SpecialEdge &)> &include,
    uint64_t &count,
    std::string &error);
bool read_regular_edge_csr_file(
    const std::string &path,
    RegularEdgeCsr &csr,
    std::string &error);
bool read_special_edge_csr_file(
    const std::string &path,
    SpecialEdgeCsr &csr,
    std::string &error);
bool convert_special_edge_csv_to_binary(const std::string &csv_path,
                                        const std::string &binary_path,
                                        uint64_t &count,
                                        std::string &error);

} // namespace ANNS

#endif // ANNS_UNG_SPECIAL_EDGE_IO_H
