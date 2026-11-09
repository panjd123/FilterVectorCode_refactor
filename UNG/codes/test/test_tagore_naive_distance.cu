#include "../src/tagore_naive_distance.cuh"
#include "../include/tagore_graph_builder.h"
#include <cuda_runtime.h>
#ifdef ANNS_TEST_COMPARE_TAGORE
#include <Tagore_src.cuh>
#endif
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
constexpr unsigned points = 64, capacity = 128;
constexpr float infinity = 10000000.0f;
using Edge = std::pair<unsigned, float>;
void verify(unsigned dim, unsigned nn, unsigned no, bool original = false) {
    std::vector<half> data(points * dim);
    std::vector<float> powers(points, 0.0f);
    unsigned state = 7231 + dim;
    for (unsigned i = 0; i < points; ++i)
        for (unsigned d = 0; d < dim; ++d) {
            state = state * 1664525u + 1013904223u;
            const float value = (16 + ((state >> 16) % 64)) / 64.0f;
            data[i * dim + d] = __float2half(value);
            powers[i] += value * value;
        }
    std::vector<unsigned> hybrid(128, 0), counts(points, 0), ids(points * capacity);
    std::vector<float> distances(points * capacity);
    for (unsigned j = 0; j < nn; ++j) hybrid[j] = (j * 17 + 3) % points;
    for (unsigned j = 0; j < no; ++j) hybrid[64 + j] = (j * 13 + 7) % points;
    std::vector<std::vector<Edge>> expected(points);
    auto distance = [&](unsigned a, unsigned b) {
        float dot = 0;
        for (unsigned d = 0; d < dim; ++d)
            dot = std::fma(__half2float(data[a * dim + d]), __half2float(data[b * dim + d]), dot);
        if (dot < .01f) dot = -infinity;
        const float value = powers[a] + powers[b] - 2 * dot;
        return value < .01f ? infinity : value;
    };
    auto minimum = [&](unsigned source, unsigned offset, unsigned count, unsigned tile) {
        Edge best{0, 3.402823466e+38F};
        for (unsigned j = tile * 16; j < tile * 16 + 16; ++j) {
            const unsigned target = j < count ? hybrid[offset + j] : 0;
            const float value = j < count ? distance(source, target) : powers[source] + powers[0] + 2 * infinity;
            if (value <= best.second) best = {target, value};
        }
        return best;
    };
    for (unsigned row = 0; row < nn; ++row)
        for (unsigned tile = 0; tile < 4; ++tile) {
            const unsigned source = hybrid[row];
            if (no) { const auto e = minimum(source, 64, no, tile); if (e.second < infinity) expected[source].push_back(e); }
            if (row / 16 <= tile) expected[source].push_back(minimum(source, 0, nn, tile));
        }
    for (unsigned row = 0; row < no; ++row)
        for (unsigned tile = 0; tile < 4; ++tile) {
            const unsigned source = hybrid[64 + row];
            const auto e = minimum(source, 0, nn, tile);
            if (e.second < infinity) expected[source].push_back(e);
        }
    half *device_data; float *device_power, *device_dist;
    unsigned *device_hybrid, *device_counts, *device_ids, *device_nn, *device_no;
    check(cudaMalloc(&device_data, data.size() * sizeof(half)));
    check(cudaMalloc(&device_power, powers.size() * sizeof(float)));
    check(cudaMalloc(&device_dist, distances.size() * sizeof(float)));
    check(cudaMalloc(&device_hybrid, hybrid.size() * sizeof(unsigned)));
    check(cudaMalloc(&device_counts, counts.size() * sizeof(unsigned)));
    check(cudaMalloc(&device_ids, ids.size() * sizeof(unsigned)));
    check(cudaMalloc(&device_nn, sizeof(unsigned))); check(cudaMalloc(&device_no, sizeof(unsigned)));
    check(cudaMemcpy(device_data, data.data(), data.size()*sizeof(half), cudaMemcpyHostToDevice));
    check(cudaMemcpy(device_power, powers.data(), powers.size()*sizeof(float), cudaMemcpyHostToDevice));
    check(cudaMemcpy(device_hybrid, hybrid.data(), hybrid.size()*sizeof(unsigned), cudaMemcpyHostToDevice));
    check(cudaMemcpy(device_nn, &nn, sizeof(unsigned), cudaMemcpyHostToDevice));
    check(cudaMemcpy(device_no, &no, sizeof(unsigned), cudaMemcpyHostToDevice));
    check(cudaMemset(device_counts, 0, counts.size()*sizeof(unsigned)));
#ifdef ANNS_TEST_COMPARE_TAGORE
    if (original)
        nn_descent_opt_cal<<<1, dim3(32,16)>>>(nullptr, device_ids, device_data, device_power, 0,
            device_dist, device_counts, device_nn, device_no, device_hybrid, dim, 96);
    else
#endif
        ANNS::tagore_naive_local_join<<<1, dim3(32,16)>>>(device_data, device_power, device_ids, device_dist,
            device_counts, device_nn, device_no, device_hybrid, dim);
    check(cudaGetLastError()); check(cudaDeviceSynchronize());
    check(cudaMemcpy(counts.data(), device_counts, counts.size()*sizeof(unsigned), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(ids.data(), device_ids, ids.size()*sizeof(unsigned), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(distances.data(), device_dist, distances.size()*sizeof(float), cudaMemcpyDeviceToHost));
    for (unsigned i = 0; i < points; ++i) {
        if (counts[i] != expected[i].size()) throw std::runtime_error("candidate count differs from CPU reference");
        std::vector<Edge> actual;
        for (unsigned j = 0; j < counts[i]; ++j) actual.emplace_back(ids[i*capacity+j], distances[i*capacity+j]);
        std::sort(actual.begin(), actual.end()); std::sort(expected[i].begin(), expected[i].end());
        for (unsigned j = 0; j < counts[i]; ++j)
            if (actual[j].first != expected[i][j].first || std::abs(actual[j].second-expected[i][j].second) > .002f)
                throw std::runtime_error("candidate or distance differs from CPU reference");
    }
    cudaFree(device_data); cudaFree(device_power); cudaFree(device_dist); cudaFree(device_hybrid);
    cudaFree(device_counts); cudaFree(device_ids); cudaFree(device_nn); cudaFree(device_no);
}
}
int main() {
    try {
        unsetenv("UNG_TAGORE_NAIVE"); setenv("UNG_TAGORE_BATCH_STREAMS", "4", 1);
        setenv("UNG_TAGORE_OPTIMIZED_TOPK", "1", 1); setenv("UNG_TAGORE_COMPACT_D2H", "1", 1);
        const auto original = ANNS::make_tagore_cuda_runtime_config(96,32,true);
        if (original.naive_distance || original.requested_streams != 4 || !original.optimized_topk || !original.compact_d2h_requested)
            throw std::runtime_error("existing defaults changed");
        setenv("UNG_TAGORE_NAIVE", "1", 1);
        const auto naive = ANNS::make_tagore_cuda_runtime_config(96,32,true);
        if (!naive.naive_distance || naive.requested_streams != 1 || naive.optimized_topk || naive.compact_d2h_requested || naive.fast_exact_batch_threshold)
            throw std::runtime_error("naive contract not enforced");
        unsigned cases = 0;
        for (unsigned dim : {32u,129u,768u})
            for (unsigned nn : {0u,1u,17u,64u})
                for (unsigned no : {0u,9u,64u}) {
                    verify(dim,nn,no); ++cases;
#ifdef ANNS_TEST_COMPARE_TAGORE
                    verify(dim,nn,no,true); ++cases;
#endif
                }
        std::cout << "GPU-Naive: runtime defaults/overrides and " << cases << " CPU-reference local-join cases passed.\n";
        return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
