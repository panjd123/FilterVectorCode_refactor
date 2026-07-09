#ifndef ANNS_GPU_CROSS_EDGE_COMMON_CUH
#define ANNS_GPU_CROSS_EDGE_COMMON_CUH

#include <cstdint>
#include <vector>

// Shared CUDA-side descriptors for cross-edge search routes.
// Keep POD layout stable: these structs are copied directly to device buffers.
struct UngGroupQueryDesc {
    uint32_t qi;
    uint32_t x_off;
    uint32_t nx;
};

struct UngGroupTileDesc {
    uint32_t q_start;
    uint32_t q_count;
    uint32_t x_off;
    uint32_t nx;
};

struct UngSourceQueryDesc {
    uint32_t qid;
    uint32_t edge_start;
    uint32_t edge_count;
};

struct UngSourceTileDesc {
    uint32_t q_start;
    uint32_t q_count;
    uint32_t out_start;
    uint32_t edge_start;
    uint32_t edge_count;
};

struct UngTargetSegmentDesc {
    uint32_t x_off;
    uint32_t nx;
};

struct CrossEdgeTopkOutputWriter {
    ANNS::CrossEdgeGpuWritebackMode mode = ANNS::CrossEdgeGpuWritebackMode::SearchQueue;
    bool id_only_writeback = false;
    std::vector<ANNS::SearchQueue>* search_queues = nullptr;
    std::vector<std::vector<ANNS::IdxType>>* id_vectors = nullptr;
    std::vector<ANNS::IdxType>* flat_ids = nullptr;

    bool writes_flat_ids() const
    {
        return mode == ANNS::CrossEdgeGpuWritebackMode::FlatId;
    }

    bool writes_id_vectors() const
    {
        return mode == ANNS::CrossEdgeGpuWritebackMode::IdVector;
    }

    bool writes_search_queues() const
    {
        return mode == ANNS::CrossEdgeGpuWritebackMode::SearchQueue;
    }
};

#endif // ANNS_GPU_CROSS_EDGE_COMMON_CUH
