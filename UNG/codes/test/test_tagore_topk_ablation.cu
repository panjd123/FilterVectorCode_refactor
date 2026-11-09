#include "../src/tagore_topk_ablation.cuh"

#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <map>
#include <stdexcept>
#include <vector>

namespace
{
void check(cudaError_t status)
{
   if (status != cudaSuccess)
      throw std::runtime_error(cudaGetErrorString(status));
}
struct Candidate
{
   unsigned id;
   float distance;
   bool visited;
};
}

int main()
{
   constexpr unsigned k = 96, cap = 128;
   unsigned *graph = nullptr, *reverse = nullptr, *count = nullptr;
   float *distances = nullptr, *reverse_distances = nullptr;
   bool *visited = nullptr;
   try
   {
      check(cudaMalloc(&graph, k * sizeof(unsigned)));
      check(cudaMalloc(&reverse, cap * sizeof(unsigned)));
      check(cudaMalloc(&count, sizeof(unsigned)));
      check(cudaMalloc(&distances, k * sizeof(float)));
      check(cudaMalloc(&reverse_distances, cap * sizeof(float)));
      check(cudaMalloc(&visited, k * sizeof(bool)));
      for (bool final_reverse : {false, true})
      {
         for (unsigned incoming : {0u, 1u, 13u, 128u})
         {
            std::array<unsigned, k> old_ids{};
            std::array<float, k> old_dist{};
            std::array<bool, k> old_visit{};
            std::array<unsigned, cap> new_ids{};
            std::array<float, cap> new_dist{};
            std::map<unsigned, Candidate> candidates;
            for (unsigned i = 0; i < k; ++i)
            {
               old_ids[i] = i + 1;
               old_dist[i] = 100.0f + 2 * i;
               old_visit[i] = i % 3 == 0;
               candidates.emplace(old_ids[i], Candidate{old_ids[i], old_dist[i], old_visit[i]});
            }
            for (unsigned i = 0; i < cap; ++i)
            {
               new_ids[i] = k + 1 + i;
               new_dist[i] = i + 0.5f;
               if (i % 11 == 0)
                  new_ids[i] = 3; // Existing neighbor: preserve its old distance/visit.
               else if (i % 7 == 0)
               {
                  new_ids[i] = new_ids[i - 1];
                  new_dist[i] = new_dist[i - 1];
               }
               if (i == 2)
                  new_ids[i] = 0; // Never add the source itself.
               if (i < std::min(incoming, final_reverse ? k : cap) && new_ids[i] != 0)
                  candidates.emplace(new_ids[i], Candidate{new_ids[i], new_dist[i], false});
            }
            std::vector<Candidate> expected;
            for (const auto &item : candidates)
               expected.push_back(item.second);
            std::sort(expected.begin(), expected.end(), [](const auto &a, const auto &b) {
               return a.distance < b.distance || (a.distance == b.distance && a.id < b.id);
            });
            check(cudaMemcpy(graph, old_ids.data(), sizeof(old_ids), cudaMemcpyHostToDevice));
            check(cudaMemcpy(distances, old_dist.data(), sizeof(old_dist), cudaMemcpyHostToDevice));
            check(cudaMemcpy(visited, old_visit.data(), sizeof(old_visit), cudaMemcpyHostToDevice));
            check(cudaMemcpy(reverse, new_ids.data(), sizeof(new_ids), cudaMemcpyHostToDevice));
            check(cudaMemcpy(reverse_distances, new_dist.data(), sizeof(new_dist), cudaMemcpyHostToDevice));
            check(cudaMemcpy(count, &incoming, sizeof(incoming), cudaMemcpyHostToDevice));
            if (final_reverse)
               ANNS::tagore_reverse_full_sort<<<1, 96>>>(graph, reverse, distances, reverse_distances, count, k);
            else
               ANNS::tagore_nn_descent_full_sort<<<1, 96>>>(graph, reverse, 0, 2, distances,
                                                            reverse_distances, visited, count, k);
            check(cudaGetLastError());
            check(cudaDeviceSynchronize());
            std::array<unsigned, k> result_ids{};
            std::array<float, k> result_dist{};
            std::array<bool, k> result_visit{};
            check(cudaMemcpy(result_ids.data(), graph, sizeof(result_ids), cudaMemcpyDeviceToHost));
            check(cudaMemcpy(result_dist.data(), distances, sizeof(result_dist), cudaMemcpyDeviceToHost));
            check(cudaMemcpy(result_visit.data(), visited, sizeof(result_visit), cudaMemcpyDeviceToHost));
            for (unsigned i = 0; i < k; ++i)
               if (result_ids[i] != expected[i].id || result_dist[i] != expected[i].distance ||
                   (!final_reverse && result_visit[i] != expected[i].visited))
                  throw std::runtime_error("full-sort top-k differs from CPU reference");
         }
      }
      cudaFree(graph); cudaFree(reverse); cudaFree(count);
      cudaFree(distances); cudaFree(reverse_distances); cudaFree(visited);
      std::cout << "Full-sort top-k: 8 CPU-reference cases passed.\n";
      return 0;
   }
   catch (const std::exception &error)
   {
      std::cerr << error.what() << '\n';
      cudaFree(graph); cudaFree(reverse); cudaFree(count);
      cudaFree(distances); cudaFree(reverse_distances); cudaFree(visited);
      return 1;
   }
}
