#include <omp.h>

#include "include/uni_nav_graph.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <roaring/roaring.hh>

#include "include/MethodSelector.h"

namespace ANNS
{

   void UniNavGraph::warmup_selectors(uint32_t num_threads)
   {
      // Warm both ONNX selectors before timed query execution.
      if (_trie_method_selector != nullptr)
      {
         std::cout << "- Warming up Trie Method Selector (Idea1)..." << std::endl;
         std::vector<float> dummy_features1(7, 0.0f);

#pragma omp parallel for num_threads(num_threads)
         for (int i = 0; i < num_threads; ++i)
         {
            _trie_method_selector->predict(dummy_features1);
         }
         std::cout << "- Trie Method Selector is warm." << std::endl;
      }

      if (_ung_acorn_selector != nullptr)
      {
         std::cout << "- Warming up UNG/ACORN Selector (Idea2)..." << std::endl;
         std::vector<float> dummy_features2(6, 0.0f);

#pragma omp parallel for num_threads(num_threads)
         for (int i = 0; i < num_threads; ++i)
         {
            _ung_acorn_selector->predict(dummy_features2);
         }
         std::cout << "- UNG/ACORN Selector is warm." << std::endl;
      }
   }

   std::vector<float> UniNavGraph::calculate_idea1_features(const QueryStats &stats) const
   {
      std::vector<float> features;
      const float epsilon = 1e-9f;
      const float query_size = static_cast<float>(stats.query_length);
      const float cand_size = static_cast<float>(stats.candidate_set_size);
      const float trie_nodes = static_cast<float>(stats.trie_total_nodes);
      const float trie_cardinality = static_cast<float>(stats.trie_label_cardinality);
      const float trie_branching = static_cast<float>(stats.trie_avg_branching_factor);

      // Feature order is part of the trained selector contract. Add a new
      // dataset branch or selector version instead of reordering an existing one.
      if (_dataset == "celeba")
      {
         features.reserve(7);
         features.push_back(cand_size * query_size);                                              // 1. Cand_x_Query_Interaction
         features.push_back(std::log1p(cand_size) * std::log1p(query_size));                      // 2. LogCand_x_LogQuery
         features.push_back(cand_size * cand_size);                                               // 4. CandSize_sq
         features.push_back(cand_size / (trie_nodes + epsilon));                                  // 6. Cand_Coverage_Ratio
         features.push_back(cand_size);                                                           // 5. CandSize
         features.push_back(trie_branching * cand_size);                                          // 3. Branching_x_CandSize
         features.push_back((trie_nodes / (trie_cardinality + epsilon)) / (cand_size + epsilon)); // 7. Cand_Selectivity
      }
      else if (_dataset == "bigann")
      {
         features.reserve(7);
         features.push_back(std::log1p(cand_size) * std::log1p(query_size));                      // 2. LogCand_x_LogQuery
         features.push_back(cand_size * query_size);                                              // 1. Cand_x_Query_Interaction
         features.push_back((trie_nodes / (trie_cardinality + epsilon)) / (cand_size + epsilon)); // 7. Cand_Selectivity
         features.push_back(cand_size * cand_size);                                               // 4. CandSize_sq
         features.push_back(cand_size / (trie_nodes + epsilon));                                  // 6. Cand_Coverage_Ratio
         features.push_back(cand_size);                                                           // 5. CandSize
         features.push_back(trie_branching * cand_size);                                          // 3. Branching_x_CandSize
      }
      else if (_dataset == "Genome")
      {
         features.reserve(7);
         features.push_back(cand_size * query_size);                                              // 1. Cand_x_Query_Interaction
         features.push_back(std::log1p(cand_size) * std::log1p(query_size));                      // 2. LogCand_x_LogQuery
         features.push_back((trie_nodes / (trie_cardinality + epsilon)) / (cand_size + epsilon)); // 7. Cand_Selectivity
         features.push_back(trie_branching * cand_size);                                          // 3. Branching_x_CandSize
         features.push_back(cand_size / (trie_nodes + epsilon));                                  // 6. Cand_Coverage_Ratio
         features.push_back(cand_size * cand_size);                                               // 4. CandSize_sq
         features.push_back(cand_size);                                                           // 5. CandSize
      }
      else if (_dataset == "words")
      {
         features.reserve(7);
         features.push_back(cand_size * query_size);                         // 1. Cand_x_Query_Interaction
         features.push_back(std::log1p(cand_size) * std::log1p(query_size)); // 2. LogCand_x_LogQuery
         features.push_back(query_size);                                     // 8. QuerySize
         features.push_back(query_size * trie_branching);                    // 9. Query_Path_Density
         features.push_back(query_size / (trie_cardinality + epsilon));      // 7. Query_Cardinality_Ratio
         features.push_back(trie_branching * query_size);                    // 3. Branching_x_QuerySize
         features.push_back(query_size * query_size);                        // 4. QuerySize_sq
      }
      else if (_dataset == "MTG")
      {
         features.reserve(7);
         features.push_back(cand_size * query_size);                                              // 1. Cand_x_Query_Interaction
         features.push_back(std::log1p(cand_size) * std::log1p(query_size));                      // 2. LogCand_x_LogQuery
         features.push_back((trie_nodes / (trie_cardinality + epsilon)) / (cand_size + epsilon)); // Cand_Selectivity
         features.push_back(cand_size * cand_size);                                               // 4. CandSize_sq
         features.push_back(cand_size / (trie_nodes + epsilon));                                  // 6. Cand_Coverage_Ratio
         features.push_back(cand_size);                                                           // 5. CandSize
         features.push_back(trie_branching * cand_size);                                          // 3. Branching_x_CandSize
      }
      else if (_dataset == "Reviews")
      {
         features.reserve(7);
         features.push_back(std::log1p(cand_size) * std::log1p(query_size)); // 2. LogCand_x_LogQuery
         features.push_back(query_size);                                     // 8. QuerySize
         features.push_back(query_size * trie_branching);                    // 9. Query_Path_Density
         features.push_back(query_size / (trie_cardinality + epsilon));      // 7. Query_Cardinality_Ratio
         features.push_back(trie_branching * query_size);                    // 3. Branching_x_QuerySize
         features.push_back(query_size * query_size);                        // 4. QuerySize_sq
         features.push_back(cand_size * query_size);                         // 1. Cand_x_Query_Interaction
      }
      else if (_dataset == "Amazon")
      {
         features.reserve(7);
         features.push_back(cand_size * query_size);                         // 1. Cand_x_Query_Interaction
         features.push_back(std::log1p(cand_size) * std::log1p(query_size)); // 2. LogCand_x_LogQuery
         features.push_back(query_size);                                     // 8. QuerySize
         features.push_back(query_size * trie_branching);                    // 9. Query_Path_Density
         features.push_back(query_size / (trie_cardinality + epsilon));      // 7. Query_Cardinality_Ratio
         features.push_back(trie_branching * query_size);                    // 3. Branching_x_QuerySize
         features.push_back(query_size * query_size);                        // 4. QuerySize_sq
      }
      else if (_dataset == "Music")
      {
         features.reserve(7);
         features.push_back(cand_size * query_size);                                              // 1. Cand_x_Query_Interaction
         features.push_back(std::log1p(cand_size) * std::log1p(query_size));                      // 2. LogCand_x_LogQuery
         features.push_back(query_size);                                                          // 8. QuerySize
         features.push_back(query_size * trie_branching);                                         // 9. Query_Path_Density
         features.push_back(query_size / (trie_cardinality + epsilon));                           // 7. Query_Cardinality_Ratio
         features.push_back((trie_nodes / (trie_cardinality + epsilon)) / (cand_size + epsilon)); // Cand_Selectivity
         features.push_back(cand_size / (trie_nodes + epsilon));                                  // 6. Cand_Coverage_Ratio
      }
      else if (_dataset == "openpmc")
      {
         features.reserve(7);
         features.push_back(query_size / (cand_size + epsilon));             // 7. Query_Cand_Ratio
         features.push_back(trie_branching * cand_size);                     // 3. Branching_x_CandSize
         features.push_back(cand_size * cand_size);                          // 4. CandSize_sq
         features.push_back(cand_size * query_size);                         // 1. Cand_x_Query_Interaction
         features.push_back(cand_size);                                      // 5. CandSize
         features.push_back(cand_size / (trie_nodes + epsilon));             // 6. Cand_Coverage_Ratio
         features.push_back(std::log1p(cand_size) * std::log1p(query_size)); // 2. LogCand_x_LogQuery
      }
      else if (_dataset == "AllNews")
      {
         features.reserve(7);
         features.push_back(cand_size * query_size);                                              // 1. Cand_x_Query_Interaction
         features.push_back(std::log1p(cand_size) * std::log1p(query_size));                      // 2. LogCand_x_LogQuery
         features.push_back(cand_size);                                                           // 5. CandSize
         features.push_back(trie_branching * cand_size);                                          // 3. Branching_x_CandSize
         features.push_back(cand_size / (trie_nodes + epsilon));                                  // 6. Cand_Coverage_Ratio
         features.push_back(cand_size * cand_size);                                               // 4. CandSize_sq
         features.push_back((trie_nodes / (trie_cardinality + epsilon)) / (cand_size + epsilon)); // Cand_Selectivity
      }
      else if (_dataset == "Russian")
      {
         features.reserve(7);
         features.push_back(cand_size * cand_size);                                               // 4. CandSize_sq
         features.push_back(trie_branching * cand_size);                                          // 3. Branching_x_CandSize
         features.push_back(cand_size);                                                           // 5. CandSize
         features.push_back(cand_size / (trie_nodes + epsilon));                                  // 6. Cand_Coverage_Ratio
         features.push_back((trie_nodes / (trie_cardinality + epsilon)) / (cand_size + epsilon)); // Cand_Selectivity
         features.push_back(query_size / (cand_size + epsilon));                                  // 7. Query_Cand_Ratio
         features.push_back(cand_size * query_size);                                              // 1. Cand_x_Query_Interaction
      }
      else if (_dataset == "VariousImg")
      {
         features.reserve(7);
         features.push_back(cand_size * query_size);                                              // 1. Cand_x_Query_Interaction
         features.push_back(std::log1p(cand_size) * std::log1p(query_size));                      // 2. LogCand_x_LogQuery
         features.push_back(cand_size);                                                           // 5. CandSize
         features.push_back(cand_size * cand_size);                                               // 4. CandSize_sq
         features.push_back(cand_size / (trie_nodes + epsilon));                                  // 6. Cand_Coverage_Ratio
         features.push_back(trie_branching * cand_size);                                          // 3. Branching_x_CandSize
         features.push_back((trie_nodes / (trie_cardinality + epsilon)) / (cand_size + epsilon)); // Cand_Selectivity
      }
      else if (_dataset == "Tiktok")
      {
         features.reserve(7);
         features.push_back(cand_size * query_size);                                              // 1. Cand_x_Query_Interaction
         features.push_back(std::log1p(cand_size) * std::log1p(query_size));                      // 2. LogCand_x_LogQuery
         features.push_back((trie_nodes / (trie_cardinality + epsilon)) / (cand_size + epsilon)); // Cand_Selectivity
         features.push_back(cand_size / (trie_nodes + epsilon));                                  // 6. Cand_Coverage_Ratio
         features.push_back(cand_size);                                                           // 5. CandSize
         features.push_back(cand_size * cand_size);                                               // 4. CandSize_sq
         features.push_back(trie_branching * cand_size);                                          // 3. Branching_x_CandSize
      }
      else if (_dataset == "cord_19")
      {
         features.reserve(7);
         features.push_back(std::log1p(cand_size) * std::log1p(query_size));                      // 2. LogCand_x_LogQuery
         features.push_back(cand_size * query_size);                                              // 1. Cand_x_Query_Interaction
         features.push_back(cand_size * cand_size);                                               // 4. CandSize_sq
         features.push_back(cand_size / (trie_nodes + epsilon));                                  // 6. Cand_Coverage_Ratio
         features.push_back(trie_branching * cand_size);                                          // 3. Branching_x_CandSize
         features.push_back(cand_size);                                                           // 5. CandSize
         features.push_back((trie_nodes / (trie_cardinality + epsilon)) / (cand_size + epsilon)); // Cand_Selectivity
      }
      else if (_dataset == "Laion")
      {
         features.reserve(7);
         features.push_back(cand_size / (trie_nodes + epsilon));                                  // 6. Cand_Coverage_Ratio
         features.push_back(cand_size * cand_size);                                               // 4. CandSize_sq
         features.push_back(cand_size);                                                           // 5. CandSize
         features.push_back(cand_size * query_size);                                              // 1. Cand_x_Query_Interaction
         features.push_back(trie_branching * cand_size);                                          // 3. Branching_x_CandSize
         features.push_back((trie_nodes / (trie_cardinality + epsilon)) / (cand_size + epsilon)); // Cand_Selectivity
         features.push_back(std::log1p(cand_size) * std::log1p(query_size));                      // 2. LogCand_x_LogQuery
      }
      else if (_dataset == "BookReviews")
      {
         features.reserve(7);
         features.push_back(cand_size * query_size);                                              // 1. Cand_x_Query_Interaction
         features.push_back(std::log1p(cand_size) * std::log1p(query_size));                      // 2. LogCand_x_LogQuery
         features.push_back(cand_size * cand_size);                                               // 4. CandSize_sq
         features.push_back(cand_size / (trie_nodes + epsilon));                                  // 6. Cand_Coverage_Ratio
         features.push_back(cand_size);                                                           // 5. CandSize
         features.push_back(trie_branching * cand_size);                                          // 3. Branching_x_CandSize
         features.push_back((trie_nodes / (trie_cardinality + epsilon)) / (cand_size + epsilon)); // Cand_Selectivity
      }
      else
      {
         std::cerr << "Warning: Dataset name not recognized for Idea1 feature calculation." << std::endl;
      }
      return features;
   }

   std::vector<float> UniNavGraph::calculate_idea2_features(const QueryStats &stats) const
   {
      std::vector<float> features;
      const float epsilon = 1e-9f;
      const float num_descendants = static_cast<float>(stats.num_lng_descendants);
      const float total_coverage = stats.entry_group_total_coverage;
      const float num_entries = static_cast<float>(stats.num_entry_points);

      // Feature order is part of the trained selector contract. Add a new
      // dataset branch or selector version instead of reordering an existing one.
      if (_dataset == "celeba")
      {
         features.reserve(6);
         features.push_back(total_coverage * total_coverage);   // 2. CoverageSquared
         features.push_back(num_descendants * total_coverage);  // 1. DescCovInteraction
         features.push_back(total_coverage);                    // 6. TotalCoverage
         features.push_back(std::log1p(total_coverage));        // 3. LogTotalCoverage
         features.push_back(std::log1p(num_descendants));       // 4. LogNumDescendants
         features.push_back(num_descendants * num_descendants); // 5. DescendantsSquared
      }
      else if (_dataset == "bigann")
      {
         features.reserve(6);
         features.push_back(std::log1p(total_coverage));        // 3. LogTotalCoverage
         features.push_back(total_coverage * total_coverage);   // 2. CoverageSquared
         features.push_back(std::log1p(num_descendants));       // 4. LogNumDescendants
         features.push_back(total_coverage);                    // 6. TotalCoverage
         features.push_back(num_descendants * total_coverage);  // 1. DescCovInteraction
         features.push_back(num_descendants * num_descendants); // 5. DescendantsSquared
      }
      else if (_dataset == "Genome")
      {
         features.reserve(6);
         features.push_back(total_coverage * total_coverage);  // CoverageSquared
         features.push_back(total_coverage);                   // TotalCoverage
         features.push_back(std::log1p(total_coverage));       // LogTotalCoverage
         features.push_back(std::log1p(num_descendants));      // LogNumDescendants
         features.push_back(num_descendants * total_coverage); // DescCovInteraction
         features.push_back(num_entries * num_descendants);    // EntriesDescInteraction
      }
      else if (_dataset == "words")
      {
         features.reserve(6);
         features.push_back(num_descendants * total_coverage);             // DescCovInteraction
         features.push_back(num_entries * num_descendants);                // EntriesDescInteraction
         features.push_back(num_entries * total_coverage);                 // EntriesCovInteraction
         features.push_back(std::log1p(total_coverage));                   // LogTotalCoverage
         features.push_back(total_coverage * total_coverage);              // CoverageSquared
         features.push_back(total_coverage / (num_entries + epsilon));     // CovPerEntry
      }
      else if (_dataset == "MTG")
      {
         features.reserve(6);
         features.push_back(std::log1p(num_entries));          // 5. LogNumEntries
         features.push_back(total_coverage * total_coverage);  // CoverageSquared
         features.push_back(num_entries);                      // 4. NumEntries
         features.push_back(std::log1p(total_coverage));       // 2. LogTotalCoverage
         features.push_back(num_entries * total_coverage);     // EntriesCovInteraction
         features.push_back(total_coverage);                   // 3. TotalCoverage
      }
      else if (_dataset == "Reviews")
      {
         features.reserve(6);
         features.push_back(total_coverage * total_coverage);              // CoverageSquared
         features.push_back(std::log1p(total_coverage));                   // 2. LogTotalCoverage
         features.push_back(total_coverage);                               // 3. TotalCoverage
         features.push_back(total_coverage / (num_entries + epsilon));     // CovPerEntry
         features.push_back(num_descendants * total_coverage);             // DescCovInteraction
         features.push_back(std::log1p(num_descendants));                  // 4. LogNumDescendants
      }
      else if (_dataset == "Amazon")
      {
         features.reserve(6);
         features.push_back(total_coverage * total_coverage);   // CoverageSquared
         features.push_back(num_descendants * total_coverage);  // 1. DescCovInteraction
         features.push_back(std::log1p(num_descendants));       // 4. LogNumDescendants
         features.push_back(num_descendants * num_descendants); // 5. DescendantsSquared
         features.push_back(std::log1p(total_coverage));        // 2. LogTotalCoverage
         features.push_back(num_descendants);                   // NumDescendants
      }
      else if (_dataset == "Music")
      {
         features.reserve(6);
         features.push_back(total_coverage * total_coverage);   // CoverageSquared
         features.push_back(std::log1p(num_descendants));       // 4. LogNumDescendants
         features.push_back(std::log1p(total_coverage));        // 2. LogTotalCoverage
         features.push_back(num_descendants * num_descendants); // 5. DescendantsSquared
         features.push_back(num_descendants);                   // NumDescendants
         features.push_back(total_coverage);                    // 3. TotalCoverage
      }
      else if (_dataset == "openpmc")
      {
         features.reserve(6);
         features.push_back(total_coverage * total_coverage);  // CoverageSquared
         features.push_back(std::log1p(total_coverage));       // 2. LogTotalCoverage
         features.push_back(total_coverage);                   // 3. TotalCoverage
         features.push_back(num_descendants * total_coverage); // 1. DescCovInteraction
         features.push_back(num_entries);                      // 4. NumEntries
         features.push_back(num_entries * num_descendants);    // 6. EntriesDescInteraction
      }
      else if (_dataset == "AllNews")
      {
         features.reserve(6);
         features.push_back(std::log1p(num_descendants));       // 4. LogNumDescendants
         features.push_back(total_coverage * total_coverage);   // CoverageSquared
         features.push_back(num_descendants * num_descendants); // 5. DescendantsSquared
         features.push_back(std::log1p(total_coverage));        // 2. LogTotalCoverage
         features.push_back(num_descendants);                   // NumDescendants
         features.push_back(num_descendants * total_coverage);  // 1. DescCovInteraction
      }
      else if (_dataset == "Russian")
      {
         features.reserve(6);
         features.push_back(num_descendants * total_coverage); // DescCovInteraction
         features.push_back(total_coverage * total_coverage);  // CoverageSquared
         features.push_back(num_entries * total_coverage);     // EntriesCovInteraction
         features.push_back(std::log1p(total_coverage));       // LogTotalCoverage
         features.push_back(total_coverage);                   // TotalCoverage
         features.push_back(num_entries);                      // NumEntries
      }
      else if (_dataset == "VariousImg")
      {
         features.reserve(6);
         features.push_back(num_descendants * total_coverage);              // 1. DescCovInteraction
         features.push_back(std::log1p(total_coverage));                    // 2. LogTotalCoverage
         features.push_back(total_coverage * total_coverage);               // CoverageSquared
         features.push_back(std::log1p(num_descendants));                   // 4. LogNumDescendants
         features.push_back(num_descendants * num_descendants);             // 5. DescendantsSquared
         features.push_back(num_descendants / (num_entries + epsilon));     // DescPerEntry
      }
      else if (_dataset == "Tiktok")
      {
         features.reserve(6);
         features.push_back(total_coverage * total_coverage);   // CoverageSquared
         features.push_back(std::log1p(total_coverage));        // 2. LogTotalCoverage
         features.push_back(total_coverage);                    // TotalCoverage
         features.push_back(num_descendants * num_descendants); // 5. DescendantsSquared
         features.push_back(std::log1p(num_descendants));       // 4. LogNumDescendants
         features.push_back(num_descendants * total_coverage);  // 1. DescCovInteraction
      }
      else if (_dataset == "cord_19")
      {
         features.reserve(6);
         features.push_back(num_descendants * num_descendants); // 5. DescendantsSquared
         features.push_back(std::log1p(num_descendants));       // 4. LogNumDescendants
         features.push_back(num_descendants);                   // NumDescendants
         features.push_back(num_descendants * total_coverage);  // 1. DescCovInteraction
         features.push_back(std::log1p(total_coverage));        // 2. LogTotalCoverage
         features.push_back(total_coverage * total_coverage);   // CoverageSquared
      }
      else if (_dataset == "Laion")
      {
         features.reserve(6);
         features.push_back(num_descendants * total_coverage);  // 1. DescCovInteraction
         features.push_back(num_descendants * num_descendants); // 5. DescendantsSquared
         features.push_back(std::log1p(num_descendants));       // 4. LogNumDescendants
         features.push_back(num_descendants);                   // NumDescendants
         features.push_back(total_coverage * total_coverage);   // CoverageSquared
         features.push_back(total_coverage);                    // TotalCoverage
      }
      else if (_dataset == "BookReviews")
      {
         features.reserve(6);
         features.push_back(total_coverage * total_coverage);   // CoverageSquared
         features.push_back(std::log1p(total_coverage));        // 2. LogTotalCoverage
         features.push_back(total_coverage);                    // TotalCoverage
         features.push_back(std::log1p(num_descendants));       // 4. LogNumDescendants
         features.push_back(num_descendants * num_descendants); // 5. DescendantsSquared
         features.push_back(num_descendants * total_coverage);  // 1. DescCovInteraction
      }
      else
      {
         std::cerr << "Warning: Dataset name not recognized for Idea2 feature calculation." << std::endl;
      }

      return features;
   }

   void UniNavGraph::populate_entry_group_route_stats(const std::vector<IdxType> &entry_group_ids,
                                                      QueryStats &stats) const
   {
      if (entry_group_ids.empty())
      {
         stats.num_entry_points = 0;
         stats.num_lng_descendants = 0;
         stats.entry_group_matched_points = 0;
         stats.entry_group_total_coverage = 0.0f;
         return;
      }

      stats.num_entry_points = entry_group_ids.size();

      roaring::Roaring descendants;
      roaring::Roaring coverage;
      for (auto group_id : entry_group_ids)
      {
         if (group_id <= 0 || group_id > _num_groups)
            continue;
         descendants |= _lng_descendants_rb[group_id];
         coverage |= _covered_sets_rb[group_id];
      }

      stats.num_lng_descendants = descendants.cardinality();
      stats.entry_group_matched_points = coverage.cardinality();
      stats.entry_group_total_coverage = static_cast<float>(stats.entry_group_matched_points) / _num_points;
   }

   void UniNavGraph::calculate_query_features_only(std::shared_ptr<IStorage> query_storage,
                                                   uint32_t num_threads,
                                                   const std::string &output_csv_path,
                                                   bool is_new_trie_method,
                                                   bool is_rec_more_start)
   {
      std::cout << "Starting unified feature calculation (Monitor: Label [1] Completion)..." << std::endl;
      auto num_queries = query_storage->get_num_points();
      std::vector<QueryStats> query_stats(num_queries);

      const double slow_query_threshold_ms = 500000.0;
      size_t log_interval = std::max(static_cast<size_t>(1), static_cast<size_t>(num_queries * 0.05));
      std::atomic<size_t> processed_count{0};

      omp_set_num_threads(num_threads);
#pragma omp parallel for schedule(dynamic, 1)
      for (auto id = 0; id < num_queries; ++id)
      {
         auto &stats = query_stats[id];
         const auto &query_labels = query_storage->get_label_set(id);

         stats.query_length = query_labels.size();
         if (!query_labels.empty())
         {
            stats.candidate_set_size = _trie_index.get_candidate_count_for_label(query_labels.back());
         }
         const auto &trie_metrics = _trie_static_metrics;
         stats.trie_total_nodes = trie_metrics.total_nodes;
         stats.trie_label_cardinality = trie_metrics.label_cardinality;
         stats.trie_avg_path_length = trie_metrics.avg_path_length;
         stats.trie_avg_branching_factor = trie_metrics.avg_branching_factor;

         std::vector<IdxType> entry_group_ids;
         static std::atomic<int> temp_counter{0};

         auto future_result = std::async(std::launch::async, [&]() {
            get_min_super_sets_debug(query_labels, entry_group_ids, false, true,
                                     temp_counter, is_new_trie_method, is_rec_more_start, stats, false);
         });

         std::future_status status =
             future_result.wait_for(std::chrono::milliseconds(static_cast<long long>(slow_query_threshold_ms)));
         if (status == std::future_status::timeout)
         {
#pragma omp critical
            {
               auto now = std::chrono::system_clock::now();
               std::time_t now_c = std::chrono::system_clock::to_time_t(now);
               std::cout << "\n[SLOW QUERY DETECTED - STILL RUNNING] "
                         << std::put_time(std::localtime(&now_c), "%H:%M:%S")
                         << " | ID: " << id << std::endl;

               std::cout << "  - Labels (" << query_labels.size() << "): [ ";
               for (size_t i = 0; i < query_labels.size(); ++i)
               {
                  std::cout << query_labels[i] << (i < query_labels.size() - 1 ? ", " : "");
                  if (i >= 10)
                  {
                     std::cout << "... ";
                     break;
                  }
               }
               std::cout << " ]" << std::endl;
            }
         }

         future_result.get();

         if (query_labels.size() == 1 && query_labels[0] == 1)
         {
#pragma omp critical
            {
               auto now = std::chrono::system_clock::now();
               std::time_t now_c = std::chrono::system_clock::to_time_t(now);

               std::cout << "[Monitor] \033[1;32mTarget Query Finished\033[0m: ID " << id
                         << " with Label [1] at " << std::put_time(std::localtime(&now_c), "%H:%M:%S")
                         << " (Entry Groups: " << entry_group_ids.size() << ")"
                         << std::endl;
            }
         }

         populate_entry_group_route_stats(entry_group_ids, stats);

         size_t current_processed = processed_count.fetch_add(1, std::memory_order_relaxed) + 1;
         if (current_processed % log_interval == 0 || current_processed == num_queries)
         {
#pragma omp critical
            {
               float progress = static_cast<float>(current_processed) / num_queries * 100.0f;
               auto now = std::chrono::system_clock::now();
               std::time_t now_c = std::chrono::system_clock::to_time_t(now);
               std::cout << "[Progress " << std::put_time(std::localtime(&now_c), "%H:%M:%S") << "] "
                         << current_processed << " / " << num_queries
                         << " (" << std::fixed << std::setprecision(1) << progress << "%)"
                         << std::endl;
            }
         }
      }

      std::cout << "In-memory feature calculation finished for " << num_queries << " queries." << std::endl;
      std::cout << "Now writing all features to CSV file: " << output_csv_path << std::endl;
      std::ofstream outfile(output_csv_path);
      if (!outfile.is_open())
      {
         std::cerr << "FATAL ERROR: Could not open file to save features: " << output_csv_path << std::endl;
         return;
      }

      outfile << "QueryID,QuerySize,CandSize,TrieTotalNodes,TrieLabelCardinality,TrieAvgBranchingFactor,"
              << "NumEntries,NumDescendants,TotalCoverage\n";

      for (size_t i = 0; i < query_stats.size(); ++i)
      {
         const auto &stats = query_stats[i];

         outfile << i << ","
                 << stats.query_length << ","
                 << stats.candidate_set_size << ","
                 << stats.trie_total_nodes << ","
                 << stats.trie_label_cardinality << ","
                 << stats.trie_avg_branching_factor << ","
                 << stats.num_entry_points << ","
                 << stats.num_lng_descendants << ","
                 << stats.entry_group_total_coverage << "\n";
      }

      outfile.close();
      std::cout << "Successfully saved all features to " << output_csv_path << std::endl;
   }

   std::optional<bool> UniNavGraph::check_pre_trie_heuristic(const std::string &dataset_name,
                                                            size_t query_length,
                                                            size_t candidate_set_size) const
   {
      (void)candidate_set_size;
      if (query_length <= 1)
      {
         return true;
      }

      if (dataset_name == "Russian" && query_length <= 2)
      {
         return true;
      }

      return std::nullopt;
   }

   size_t UniNavGraph::get_candidate_count_for_label(LabelType label) const
   {
      return _trie_index.get_candidate_count_for_label(label);
   }

} // namespace ANNS
