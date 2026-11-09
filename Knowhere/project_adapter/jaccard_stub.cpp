#include <stdexcept>

#include <faiss/cppcontrib/knowhere/utils/distances.h>

namespace faiss::cppcontrib::knowhere {

void knn_jaccard(
    const float*, const float*, std::size_t, std::size_t, std::size_t,
    float_maxheap_array_t*, const IDSelector*) {
    throw std::runtime_error("the standalone Milvus HNSW adapter does not support Jaccard distance");
}

}  // namespace faiss::cppcontrib::knowhere
