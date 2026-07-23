#include <iostream>
#include <fstream>
#include <random>
#include <cmath>
#include <queue>
#include <vector>
#include <string>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <set>
#include <memory>
#include <algorithm>
#include <map>
#include <cstdint>
#include <stdexcept>
#include <atomic>
#include <boost/program_options.hpp>
#include <omp.h>

// ANNS Namespace contains the core Trie data structure for indexing label sets.
namespace ANNS
{
   using IdxType = uint32_t;
   using LabelType = uint32_t;

   struct TrieNode
   {
      std::vector<std::pair<LabelType, std::unique_ptr<TrieNode>>> children;
      IdxType group_id = 0;
      IdxType group_size = 0;
      std::vector<LabelType> label_set;
      std::atomic<IdxType> superset_count{0};
      IdxType subtree_group_size_sum = 0;
   };

   class TrieIndex
   {
   public:
      TrieIndex() : root(std::make_unique<TrieNode>()), max_group_id(0), max_label_id(0) {}

      IdxType insert(const std::vector<LabelType> &labels, IdxType &new_group_id_counter)
      {
         auto node = root.get();
         std::vector<LabelType> sorted_labels = labels;
         std::sort(sorted_labels.begin(), sorted_labels.end());

         for (auto label : sorted_labels)
         {
            auto it = std::lower_bound(node->children.begin(), node->children.end(), label,
                                       [](const auto &child, LabelType val)
                                       {
                                          return child.first < val;
                                       });

            if (it == node->children.end() || it->first != label)
            {
               auto new_node = std::make_unique<TrieNode>();
               it = node->children.insert(it, {label, std::move(new_node)});
            }
            node = it->second.get();
         }

         if (node->group_id == 0)
         {
            node->group_id = new_group_id_counter++;
            node->label_set = sorted_labels;
            max_group_id = std::max(max_group_id, node->group_id);
         }
         node->group_size++;
         if (!sorted_labels.empty())
         {
            max_label_id = std::max(max_label_id, sorted_labels.back());
         }
         return node->group_id;
      }

      IdxType calculate_coverage(const std::vector<LabelType> &query_labels) const
      {
         if (query_labels.empty())
            return 0;

         std::vector<LabelType> sorted_labels = query_labels;
         std::sort(sorted_labels.begin(), sorted_labels.end());

         IdxType total_coverage = 0;
         std::set<IdxType> visited_groups;

         find_and_sum_supersets(root.get(), sorted_labels, 0, total_coverage, visited_groups);

         return total_coverage;
      }

      LabelType get_max_label_id() const { return max_label_id; }
      IdxType get_max_group_id() const { return max_group_id; }

      const TrieNode *find_node(const std::vector<LabelType> &sorted_labels) const
      {
         auto node = root.get();
         for (auto label : sorted_labels)
         {
            auto it = std::lower_bound(node->children.begin(), node->children.end(), label,
                                       [](const auto &child, LabelType val)
                                       { return child.first < val; });
            if (it == node->children.end() || it->first != label)
            {
               return nullptr;
            }
            node = it->second.get();
         }
         return node;
      }

      std::vector<TrieNode *> get_all_nodes_with_group_id()
      {
         std::vector<TrieNode *> nodes;
         std::queue<TrieNode *> q;
         q.push(root.get());
         while (!q.empty())
         {
            TrieNode *current = q.front();
            q.pop();
            if (current->group_id > 0)
            {
               nodes.push_back(current);
            }
            for (auto &child_pair : current->children)
            {
               q.push(child_pair.second.get());
            }
         }
         return nodes;
      }

      bool load_superset_cache(const std::string &cache_path)
      {
         std::ifstream cache_file(cache_path);
         if (!cache_file)
            return false;

         std::cout << "Loading superset counts from cache: " << cache_path << "..." << std::endl;
         std::string line, label_str;

         while (std::getline(cache_file, line))
         {
            std::stringstream ss(line);
            IdxType count;
            ss >> count;
            ss.ignore();

            std::vector<LabelType> label_set;
            while (std::getline(ss, label_str, ' '))
            {
               if (!label_str.empty())
               {
                  label_set.emplace_back(std::stoul(label_str));
               }
            }

            if (!label_set.empty())
            {
               TrieNode *node = const_cast<TrieNode *>(find_node(label_set));
               if (node)
               {
                  node->superset_count = count;
               }
            }
         }
         std::cout << "Cache loaded successfully." << std::endl;
         return true;
      }

      void save_superset_cache(const std::string &cache_path, const std::vector<TrieNode *> &nodes)
      {
         std::cout << "Saving superset counts to cache: " << cache_path << "..." << std::endl;
         std::ofstream cache_file(cache_path);
         if (!cache_file)
         {
            std::cerr << "Warning: Could not open cache file for writing: " << cache_path << std::endl;
            return;
         }

         for (const auto *node : nodes)
         {
            cache_file << node->superset_count.load() << ",";
            for (size_t i = 0; i < node->label_set.size(); ++i)
            {
               cache_file << node->label_set[i] << (i == node->label_set.size() - 1 ? "" : " ");
            }
            cache_file << "\n";
         }
         std::cout << "Cache saved successfully." << std::endl;
      }

      void calculate_superset_counts()
      {
         std::cout << "Starting superset count pre-computation..." << std::endl;
         std::vector<TrieNode *> nodes = get_all_nodes_with_group_id();
         size_t num_nodes = nodes.size();
         std::cout << "Found " << num_nodes << " unique label sets to process." << std::endl;

#pragma omp parallel for schedule(dynamic)
         for (size_t i = 0; i < num_nodes; ++i)
         {
            TrieNode *node_a = nodes[i];
            for (size_t j = 0; j < num_nodes; ++j)
            {
               if (i == j)
                  continue;
               TrieNode *node_b = nodes[j];
               if (node_a->label_set.size() < node_b->label_set.size())
               {
                  if (std::includes(node_b->label_set.begin(), node_b->label_set.end(),
                                    node_a->label_set.begin(), node_a->label_set.end()))
                  {
                     node_a->superset_count++;
                  }
               }
            }
            if (i > 0 && i % 1000 == 0)
            {
#pragma omp critical
               {
                  std::cout << "\rPre-computation progress: " << i << " / " << num_nodes << std::flush;
               }
            }
         }
         std::cout << "\nSuperset count pre-computation finished." << std::endl;
      }

      void precompute_subtree_sums()
      {
         calculate_subtree_sums(root.get());
      }

   private:
      IdxType calculate_subtree_sums(TrieNode *node)
      {
         if (!node)
            return 0;
         IdxType current_sum = node->group_size;
         for (auto &child_pair : node->children)
         {
            current_sum += calculate_subtree_sums(child_pair.second.get());
         }
         node->subtree_group_size_sum = current_sum;
         return current_sum;
      }

      void find_and_sum_supersets(
          const TrieNode *node,
          const std::vector<LabelType> &sorted_query_labels,
          size_t query_idx,
          IdxType &total_coverage,
          std::set<IdxType> &visited_groups) const
      {
         if (query_idx == sorted_query_labels.size())
         {
            collect_all_children_coverage(node, total_coverage, visited_groups);
            return;
         }

         LabelType current_query_label = sorted_query_labels[query_idx];

         for (const auto &child_pair : node->children)
         {
            const TrieNode *child_node = child_pair.second.get();
            LabelType child_label = child_pair.first;

            if (child_label < current_query_label)
            {
               find_and_sum_supersets(child_node, sorted_query_labels, query_idx, total_coverage, visited_groups);
            }
            else if (child_label == current_query_label)
            {
               find_and_sum_supersets(child_node, sorted_query_labels, query_idx + 1, total_coverage, visited_groups);
            }
         }
      }

      void collect_all_children_coverage(const TrieNode *node, IdxType &total_coverage, std::set<IdxType> & /*visited_groups*/) const
      {
         total_coverage += node->subtree_group_size_sum;
      }

   private:
      std::unique_ptr<TrieNode> root;
      LabelType max_label_id;
      IdxType max_group_id;
   };
}

namespace po = boost::program_options;
using namespace ANNS;

void load_fvecs(const std::string &filename, std::vector<std::vector<float>> &vectors, int &dimension)
{
   std::ifstream input(filename, std::ios::binary);
   if (!input)
   {
      throw std::runtime_error("Cannot open file: " + filename);
   }
   input.read(reinterpret_cast<char *>(&dimension), sizeof(int));
   if (input.gcount() != sizeof(int))
   {
      dimension = 0;
      return;
   }
   input.seekg(0);
   int d;
   while (input.read(reinterpret_cast<char *>(&d), sizeof(int)))
   {
      if (d != dimension)
      {
         throw std::runtime_error("Inconsistent vector dimensions in fvecs file.");
      }
      std::vector<float> vec(d);
      input.read(reinterpret_cast<char *>(vec.data()), d * sizeof(float));
      if (input.gcount() != d * sizeof(float))
      {
         throw std::runtime_error("Error reading vector data from fvecs file.");
      }
      vectors.push_back(vec);
   }
}

void write_fvecs(const std::string &filename, const std::vector<std::vector<float>> &vectors)
{
   if (vectors.empty())
      return;
   std::ofstream output(filename, std::ios::binary);
   if (!output)
   {
      throw std::runtime_error("Cannot open file for writing: " + filename);
   }
   int dimension = vectors[0].size();
   for (const auto &vec : vectors)
   {
      if (vec.size() != static_cast<size_t>(dimension))
      {
         throw std::runtime_error("Inconsistent vector dimensions when writing to fvecs file.");
      }
      output.write(reinterpret_cast<const char *>(&dimension), sizeof(int));
      output.write(reinterpret_cast<const char *>(vec.data()), dimension * sizeof(float));
   }
}

void run_generate_mode(const po::variables_map &vm)
{
   std::string input_file = vm["input_file"].as<std::string>();
   std::string output_file = vm["output_file"].as<std::string>();
   IdxType num_points = vm["num_points"].as<IdxType>();
   IdxType K = vm["K"].as<IdxType>();
   std::string distribution_type = vm["distribution_type"].as<std::string>();
   bool truncate_to_fixed_length = vm["truncate_to_fixed_length"].as<bool>();
   IdxType num_labels_per_query = vm["num_labels_per_query"].as<IdxType>();
   IdxType expected_num_label = vm["expected_num_label"].as<IdxType>();

   bool generate_vectors = vm.count("base_vectors_file") && vm.count("output_vectors_file");
   std::vector<std::vector<float>> base_vectors;

   if (generate_vectors)
   {
      std::string base_vectors_file = vm["base_vectors_file"].as<std::string>();
      std::cout << "Loading base vectors from " << base_vectors_file << "..." << std::endl;
      try
      {
         int dim;
         load_fvecs(base_vectors_file, base_vectors, dim);
         std::cout << "Loaded " << base_vectors.size() << " base vectors with dimension " << dim << "." << std::endl;
      }
      catch (const std::exception &e)
      {
         std::cerr << "Error loading base vectors: " << e.what() << std::endl;
         return;
      }
   }

   TrieIndex trie_index;
   std::ifstream infile(input_file);
   IdxType new_label_set_id = 1;
   std::string line, label_str;

   std::vector<std::vector<LabelType>> base_label_sets;
   while (std::getline(infile, line))
   {
      std::vector<LabelType> label_set;
      std::stringstream ss(line);
      while (std::getline(ss, label_str, ','))
         if (!label_str.empty())
            label_set.emplace_back(std::stoul(label_str));
      if (!label_set.empty())
      {
         trie_index.insert(label_set, new_label_set_id);
         base_label_sets.push_back(label_set);
      }
   }

   std::cout << "Pre-computing subtree sums for generation..." << std::endl;
   trie_index.precompute_subtree_sums();
   std::cout << "Pre-computation complete." << std::endl;

   if (generate_vectors && !base_vectors.empty() && base_vectors.size() != base_label_sets.size())
   {
      std::cerr << "Warning: The number of base label sets (" << base_label_sets.size()
                << ") does not match the number of base vectors (" << base_vectors.size()
                << "). Vector sampling might be inaccurate." << std::endl;
   }

   std::vector<std::vector<LabelType>> generated_label_sets;
   std::vector<std::vector<float>> generated_vectors;

   LabelType num_labels = trie_index.get_max_label_id();

   std::cout << "======================================" << std::endl;
   std::cout << "Generating " << num_points << " candidate queries in parallel..." << std::endl;
   if (generate_vectors)
   {
      std::cout << "Corresponding query vectors will also be generated." << std::endl;
   }

#pragma omp parallel for schedule(dynamic)
   for (IdxType i = 0; i < num_points; ++i)
   {
      std::mt19937 gen(std::random_device{}() + omp_get_thread_num());
      std::vector<LabelType> label_set;
      std::vector<float> corresponding_vector;
      bool found_valid = false;

      for (int j = 0; j < 500; ++j)
      {
         label_set.clear();
         if (truncate_to_fixed_length)
         {
            if (distribution_type == "zipf")
            {
               std::vector<LabelType> temp_set;
               for (LabelType label = 1; label <= num_labels; ++label)
               {
                  double probability = 0.7 / static_cast<double>(label);
                  std::bernoulli_distribution dist(probability);
                  if (dist(gen))
                  {
                     temp_set.push_back(label);
                  }
               }
               if (temp_set.size() >= num_labels_per_query)

               {
                  temp_set.resize(num_labels_per_query);
                  label_set = temp_set;
               }
            }
            else // 'uniform' mode now uses the robust Parent-Subset strategy
            {
               if (base_label_sets.empty())
                  break;

               std::uniform_int_distribution<> dis(0, base_label_sets.size() - 1);
               size_t parent_idx = dis(gen);
               const auto &parent_labels = base_label_sets[parent_idx];

               if (parent_labels.size() < num_labels_per_query)
               {
                  continue;
               }

               std::vector<LabelType> temp_set = parent_labels;
               std::shuffle(temp_set.begin(), temp_set.end(), gen);
               temp_set.resize(num_labels_per_query);
               label_set = temp_set;

               if (generate_vectors && !base_vectors.empty())
               {
                  corresponding_vector = base_vectors[parent_idx];
               }
            }
         }
         else
         {
            if (distribution_type == "zipf")
            {
               for (LabelType label = 1; label <= num_labels; ++label)
               {
                  double probability = 0.7 / static_cast<double>(label);
                  std::bernoulli_distribution dist(probability);
                  if (dist(gen))
                  {
                     label_set.push_back(label);
                  }
               }
               std::shuffle(label_set.begin(), label_set.end(), gen);
               if (label_set.size() > expected_num_label)
               {
                  label_set.resize(expected_num_label);
               }
            }
            else
            { // uniform
               for (LabelType label = 1; label <= num_labels; ++label)
               {
                  if (static_cast<float>(gen()) / gen.max() < static_cast<float>(expected_num_label) / num_labels)
                  {
                     label_set.push_back(label);
                  }
               }
            }
         }

         if (generate_vectors && corresponding_vector.empty() && !base_vectors.empty())
         {
            std::uniform_int_distribution<> dis(0, base_vectors.size() - 1);
            corresponding_vector = base_vectors[dis(gen)];
         }

         if (label_set.empty())
            continue;

         IdxType coverage = trie_index.calculate_coverage(label_set);

         if (truncate_to_fixed_length && distribution_type == "uniform" && num_labels_per_query == 9 && coverage > 500)
         {
            continue;
         }

         if (coverage >= K)
         {
            found_valid = true;
            break;
         }
      }

      if (!found_valid)
      {
#pragma omp critical
         {
            if (!generated_label_sets.empty())
            {
               std::uniform_int_distribution<> dis(0, generated_label_sets.size() - 1);
               size_t fallback_idx = dis(gen);
               label_set = generated_label_sets[fallback_idx];
               if (generate_vectors && generated_vectors.size() > fallback_idx)
               {
                  corresponding_vector = generated_vectors[fallback_idx];
               }
            }
         }
      }

      if (!label_set.empty())
      {
#pragma omp critical
         {
            generated_label_sets.push_back(label_set);
            if (generate_vectors && !corresponding_vector.empty())

            {
               generated_vectors.push_back(corresponding_vector);
            }

            if (generated_label_sets.size() % 100 == 0)

            {
               std::cout << "\rGenerated " << generated_label_sets.size() << " / " << num_points << " candidates..." << std::flush;
            }
         }
      }
   }

   std::ofstream outfile(output_file);
   for (const auto &ls : generated_label_sets)
   {
      std::vector<LabelType> sorted_ls = ls;
      std::sort(sorted_ls.begin(), sorted_ls.end());
      for (size_t k = 0; k < sorted_ls.size(); ++k)
      {
         outfile << sorted_ls[k] << (k == sorted_ls.size() - 1 ? "" : ",");
      }
      outfile << std::endl;
   }
   std::cout << "\nGeneration complete. " << generated_label_sets.size() << " candidate queries written to " << output_file << std::endl;

   if (generate_vectors)
   {
      std::string output_vectors_file = vm["output_vectors_file"].as<std::string>();
      try
      {
         write_fvecs(output_vectors_file, generated_vectors);
         std::cout << generated_vectors.size() << " corresponding query vectors written to " << output_vectors_file << std::endl;
      }
      catch (const std::exception &e)
      {
         std::cerr << "Error writing output vectors: " << e.what() << std::endl;
      }
   }
}

void run_analyze_mode(const po::variables_map &vm)
{
   std::string input_file = vm["input_file"].as<std::string>();
   std::string candidate_file = vm["candidate_file"].as<std::string>();
   std::string profiled_output = vm["profiled_output"].as<std::string>();

   TrieIndex trie_index;
   std::ifstream infile(input_file);
   IdxType new_label_set_id = 1;
   std::string line, label_str;
   while (std::getline(infile, line))
   {
      std::vector<LabelType> label_set;
      std::stringstream ss(line);
      while (std::getline(ss, label_str, ','))
         if (!label_str.empty())
            label_set.emplace_back(std::stoul(label_str));
      if (!label_set.empty())
         trie_index.insert(label_set, new_label_set_id);
   }

   std::cout << "Pre-computing subtree sums for analysis..." << std::endl;
   trie_index.precompute_subtree_sums();
   std::cout << "Pre-computation complete." << std::endl;

   std::cout << "Reading candidate file..." << std::endl;
   std::ifstream cand_file(candidate_file);

   std::vector<std::vector<LabelType>> queries_to_process;
   while (std::getline(cand_file, line))
   {
      std::vector<LabelType> label_set;
      std::stringstream ss(line);
      while (std::getline(ss, label_str, ','))
         if (!label_str.empty())
            label_set.emplace_back(std::stoul(label_str));
      if (!label_set.empty())
      {
         queries_to_process.push_back(label_set);
      }
   }
   cand_file.close();

   size_t num_queries = queries_to_process.size();
   std::cout << "Read " << num_queries << " queries. Analyzing in parallel..." << std::endl;

   std::vector<std::pair<IdxType, std::vector<LabelType>>> results(num_queries);
   std::atomic<size_t> progress_counter{0};

#pragma omp parallel for schedule(dynamic)
   for (size_t i = 0; i < num_queries; ++i)
   {
      IdxType coverage = trie_index.calculate_coverage(queries_to_process[i]);
      results[i] = {coverage, queries_to_process[i]};

      size_t processed_count = ++progress_counter;
      if (processed_count % 100 == 0)
      {
#pragma omp critical
         {
            std::cout << "\rAnalyzed " << processed_count << " / " << num_queries << " queries..." << std::flush;
         }
      }
   }

   std::cout << "\nAnalysis complete. Writing results..." << std::endl;

   std::ofstream prof_file(profiled_output);
   prof_file << "coverage_count,labels\n";
   for (const auto &res : results)
   {
      prof_file << res.first << ",";
      std::vector<LabelType> sorted_labels = res.second;
      std::sort(sorted_labels.begin(), sorted_labels.end());
      for (size_t k = 0; k < sorted_labels.size(); ++k)
      {
         prof_file << sorted_labels[k] << (k == sorted_labels.size() - 1 ? "" : " ");
      }
      prof_file << "\n";
   }
   std::cout << "Results written to " << profiled_output << std::endl;
}

void run_sub_base_subset_mode(const po::variables_map &vm)
{
   std::string input_file = vm["input_file"].as<std::string>();
   std::string output_file = vm["output_file"].as<std::string>();
   std::string base_vectors_file = vm["base_vectors_file"].as<std::string>();
   std::string output_vectors_file = vm["output_vectors_file"].as<std::string>();
   IdxType num_points = vm["num_points"].as<IdxType>();
   IdxType query_length = vm["query-length"].as<IdxType>();
   IdxType K = vm["K"].as<IdxType>();
   IdxType max_coverage = vm["max-coverage"].as<IdxType>();
   IdxType min_children = vm["min-children"].as<IdxType>();

   std::cout << "======================================================" << std::endl;
   std::cout << "--- Running in SUB_BASE (Random Parent + Random Subset) mode ---" << std::endl;
   std::cout << "Target query length: " << query_length << std::endl;
   std::cout << "Target coverage range: [" << K << ", " << max_coverage << "]" << std::endl;
   std::cout << "Minimum children (supersets): " << min_children << std::endl;
   std::cout << "======================================================" << std::endl;

   TrieIndex trie_index;
   std::ifstream infile(input_file);
   IdxType new_label_set_id = 1;
   std::string line, label_str;

   std::vector<std::vector<float>> all_base_vectors;
   int dim;
   try
   {
      load_fvecs(base_vectors_file, all_base_vectors, dim);
   }
   catch (const std::exception &e)
   {
      std::cerr << "Fatal Error loading base vectors: " << e.what() << std::endl;
      return;
   }

   std::vector<std::vector<LabelType>> filtered_label_sets;
   std::vector<std::vector<float>> filtered_vectors;

   size_t line_index = 0;
   while (std::getline(infile, line))
   {
      if (line_index >= all_base_vectors.size())
      {
         std::cerr << "Warning: Label file has more lines than vectors. Stopping at line " << line_index << std::endl;
         break;
      }

      std::vector<LabelType> label_set;
      std::stringstream ss(line);
      while (std::getline(ss, label_str, ','))
      {
         if (!label_str.empty())
         {
            label_set.emplace_back(std::stoul(label_str));
         }
      }

      if (!label_set.empty())
      {
         trie_index.insert(label_set, new_label_set_id);
         filtered_label_sets.push_back(label_set);
         filtered_vectors.push_back(all_base_vectors[line_index]);
      }
      line_index++;
   }

   std::cout << "Successfully loaded and filtered " << filtered_label_sets.size()
             << " items with non-empty labels." << std::endl;

   std::cout << "Pre-computing subtree sums for coverage optimization..." << std::endl;
   trie_index.precompute_subtree_sums();
   std::cout << "Pre-computation complete." << std::endl;

   bool cache_loaded = false;
   if (vm.count("cache-file"))
   {
      std::string cache_path = vm["cache-file"].as<std::string>();
      if (trie_index.load_superset_cache(cache_path))
      {
         cache_loaded = true;
      }
   }

   if (!cache_loaded)
   {
      trie_index.calculate_superset_counts();
      if (vm.count("cache-file"))
      {
         std::string cache_path = vm["cache-file"].as<std::string>();
         auto nodes = trie_index.get_all_nodes_with_group_id();
         trie_index.save_superset_cache(cache_path, nodes);
      }
   }

   std::vector<size_t> parent_pool_indices;
   for (size_t i = 0; i < filtered_label_sets.size(); ++i)
   {
      if (filtered_label_sets[i].size() >= query_length)
      {
         parent_pool_indices.push_back(i);
      }
   }
   if (parent_pool_indices.empty())
   {
      std::cerr << "Error: No base label sets found with length >= " << query_length << ". Cannot generate queries." << std::endl;
      return;
   }
   std::cout << "Found " << parent_pool_indices.size() << " valid parent label sets for sampling." << std::endl;

   std::vector<std::vector<LabelType>> generated_label_sets;
   std::vector<std::vector<float>> generated_vectors;
   generated_label_sets.reserve(num_points);
   generated_vectors.reserve(num_points);

   const int TOURNAMENT_SIZE = 20;
   const int MAX_PARENT_RETRIES = 500;

   // ===== NEW CODE START: Progress Counter =====
   std::atomic<size_t> progress_counter{0};
   size_t last_reported_count = 0;
   // ===== NEW CODE END =====

#pragma omp parallel
   {
      std::vector<std::vector<LabelType>> local_label_sets;
      std::vector<std::vector<float>> local_vectors;
      std::mt19937 gen(std::random_device{}() + omp_get_thread_num());
      std::uniform_int_distribution<size_t> parent_dist(0, parent_pool_indices.size() - 1);

#pragma omp for schedule(dynamic)
      for (IdxType i = 0; i < num_points; ++i)
      {
         bool query_generated = false;
         for (int parent_retry = 0; parent_retry < MAX_PARENT_RETRIES; ++parent_retry)
         {
            size_t parent_idx = parent_pool_indices[parent_dist(gen)];
            const auto &parent_labels = filtered_label_sets[parent_idx];

            for (int t = 0; t < TOURNAMENT_SIZE; ++t)
            {
               std::vector<LabelType> candidate_subset = parent_labels;
               std::shuffle(candidate_subset.begin(), candidate_subset.end(), gen);
               candidate_subset.resize(query_length);
               std::sort(candidate_subset.begin(), candidate_subset.end());

               IdxType coverage = trie_index.calculate_coverage(candidate_subset);

               if (coverage < K || coverage > max_coverage)
               {
                  continue;
               }

               const TrieNode *node = trie_index.find_node(candidate_subset);
               IdxType children_count = (node) ? node->superset_count.load() : 0;

               if (children_count >= min_children)
               {
                  local_label_sets.push_back(candidate_subset);
                  local_vectors.push_back(filtered_vectors[parent_idx]);
                  query_generated = true;
                  break;
               }
            }

            if (query_generated)
            {
               break;
            }
         }

         // ===== NEW CODE START: Update and Report Progress =====
         size_t current_progress = ++progress_counter;
         if (current_progress % 50 == 0) // Print progress every 50 tasks
         {
#pragma omp critical
            {
               // A double-check to avoid printing stale info
               if (generated_label_sets.size() > last_reported_count || current_progress % 200 == 0)
               {
                  std::cout << "\r[Progress] Tasks completed: " << current_progress << "/" << num_points
                            << ". Queries generated: " << generated_label_sets.size() << "..." << std::flush;
                  last_reported_count = generated_label_sets.size();
               }
            }
         }
         // ===== NEW CODE END =====
      }

#pragma omp critical
      {
         generated_label_sets.insert(generated_label_sets.end(), local_label_sets.begin(), local_label_sets.end());
         generated_vectors.insert(generated_vectors.end(), local_vectors.begin(), local_vectors.end());
         // Final progress update upon thread completion
         std::cout << "\r[Progress] Tasks completed: " << progress_counter.load() << "/" << num_points
                   << ". Queries generated: " << generated_label_sets.size() << "..." << std::flush;
      }
   }

   size_t num_successfully_generated = generated_label_sets.size();
   if (num_successfully_generated < num_points && num_successfully_generated > 0)
   {
      std::cout << "\n[INFO] Generated " << num_successfully_generated << " unique queries. Starting smart padding to reach " << num_points << "..." << std::endl;
      size_t num_to_padd = num_points - num_successfully_generated;

      std::mt19937 gen(std::random_device{}());
      std::uniform_int_distribution<size_t> padding_labels_dist(0, num_successfully_generated - 1);
      std::uniform_int_distribution<size_t> padding_vectors_dist(0, all_base_vectors.size() - 1);

      for (size_t i = 0; i < num_to_padd; ++i)
      {
         size_t label_idx_to_copy = padding_labels_dist(gen);
         size_t vector_idx_to_sample = padding_vectors_dist(gen);

         generated_label_sets.push_back(generated_label_sets[label_idx_to_copy]);
         generated_vectors.push_back(all_base_vectors[vector_idx_to_sample]);
      }
      std::cout << "[INFO] Padded with " << num_to_padd << " queries (re-used labels, new random vectors)." << std::endl;
      std::cout << "[INFO] Total query count is now " << generated_label_sets.size() << "." << std::endl;
   }

   std::cout << "\nGeneration complete." << std::endl;
   if (generated_label_sets.size() < num_points)
   {
      std::cout << "Warning: Could only generate " << generated_label_sets.size() << " queries meeting the criteria, and padding failed because no queries were generated." << std::endl;
   }

   std::ofstream outfile(output_file);
   for (const auto &ls : generated_label_sets)
   {
      for (size_t k = 0; k < ls.size(); ++k)
      {
         outfile << ls[k] << (k == ls.size() - 1 ? "" : ",");
      }
      outfile << std::endl;
   }
   std::cout << generated_label_sets.size() << " query labels written to " << output_file << std::endl;

   write_fvecs(output_vectors_file, generated_vectors);
   std::cout << generated_vectors.size() << " query vectors written to " << output_vectors_file << std::endl;
}

void run_weighted_sub_base_mode(const po::variables_map &vm)
{
   std::string input_file = vm["input_file"].as<std::string>();
   std::string output_file = vm["output_file"].as<std::string>();
   std::string base_vectors_file = vm["base_vectors_file"].as<std::string>();
   std::string output_vectors_file = vm["output_vectors_file"].as<std::string>();
   IdxType num_points = vm["num_points"].as<IdxType>();
   IdxType query_length = vm["query-length"].as<IdxType>();
   IdxType K = vm["K"].as<IdxType>();
   IdxType max_coverage = vm["max-coverage"].as<IdxType>();
   IdxType min_children = vm["min-children"].as<IdxType>();

   std::cout << "========================================================" << std::endl;
   std::cout << "--- Running in WEIGHTED_SUB_BASE (Random Parent + Hot Subset) mode ---" << std::endl;
   std::cout << "Target query length: " << query_length << std::endl;
   std::cout << "Target coverage range: [" << K << ", " << max_coverage << "]" << std::endl;
   std::cout << "Minimum children (supersets): " << min_children << std::endl;
   std::cout << "========================================================" << std::endl;

   TrieIndex trie_index;
   std::ifstream infile(input_file);
   IdxType new_label_set_id = 1;
   std::string line, label_str;

   std::vector<std::vector<float>> all_base_vectors;
   int dim;
   try
   {
      load_fvecs(base_vectors_file, all_base_vectors, dim);
   }
   catch (const std::exception &e)
   {
      std::cerr << "Fatal Error loading base vectors: " << e.what() << std::endl;
      return;
   }

   std::vector<std::vector<LabelType>> filtered_label_sets;
   std::vector<std::vector<float>> filtered_vectors;

   size_t line_index = 0;
   while (std::getline(infile, line))
   {
      if (line_index >= all_base_vectors.size())
      {
         std::cerr << "Warning: Label file has more lines than vectors. Stopping at line " << line_index << std::endl;
         break;
      }

      std::vector<LabelType> label_set;
      std::stringstream ss(line);
      while (std::getline(ss, label_str, ','))
      {
         if (!label_str.empty())
         {
            label_set.emplace_back(std::stoul(label_str));
         }
      }

      if (!label_set.empty())
      {
         trie_index.insert(label_set, new_label_set_id);
         filtered_label_sets.push_back(label_set);
         filtered_vectors.push_back(all_base_vectors[line_index]);
      }
      line_index++;
   }

   std::cout << "Successfully loaded and filtered " << filtered_label_sets.size()
             << " items with non-empty labels." << std::endl;

   std::cout << "Pre-computing subtree sums for coverage optimization..." << std::endl;
   trie_index.precompute_subtree_sums();
   std::cout << "Pre-computation complete." << std::endl;

   bool cache_loaded = false;
   if (vm.count("cache-file"))
   {
      std::string cache_path = vm["cache-file"].as<std::string>();
      if (trie_index.load_superset_cache(cache_path))
      {
         cache_loaded = true;
      }
   }

   if (!cache_loaded)
   {
      trie_index.calculate_superset_counts();
      if (vm.count("cache-file"))
      {
         std::string cache_path = vm["cache-file"].as<std::string>();
         auto nodes = trie_index.get_all_nodes_with_group_id();
         trie_index.save_superset_cache(cache_path, nodes);
      }
   }

   std::vector<size_t> parent_pool_indices;
   for (size_t i = 0; i < filtered_label_sets.size(); ++i)
   {
      if (filtered_label_sets[i].size() >= query_length)
      {
         parent_pool_indices.push_back(i);
      }
   }
   if (parent_pool_indices.empty())
   {
      std::cerr << "Error: No base label sets found with length >= " << query_length << ". Cannot generate queries." << std::endl;
      return;
   }
   std::cout << "Found " << parent_pool_indices.size() << " valid parent label sets for sampling." << std::endl;

   std::vector<std::vector<LabelType>> generated_label_sets;
   std::vector<std::vector<float>> generated_vectors;
   generated_label_sets.reserve(num_points);
   generated_vectors.reserve(num_points);

   const int TOURNAMENT_SIZE = 20;
   const int MAX_PARENT_RETRIES = 500;

   // ===== NEW CODE START: Progress Counter =====
   std::atomic<size_t> progress_counter{0};
   size_t last_reported_count = 0;
   // ===== NEW CODE END =====

#pragma omp parallel
   {
      std::vector<std::vector<LabelType>> local_label_sets;
      std::vector<std::vector<float>> local_vectors;
      std::mt19937 gen(std::random_device{}() + omp_get_thread_num());
      std::uniform_int_distribution<size_t> parent_dist(0, parent_pool_indices.size() - 1);

#pragma omp for schedule(dynamic)
      for (IdxType i = 0; i < num_points; ++i)
      {
         bool query_generated = false;
         for (int parent_retry = 0; parent_retry < MAX_PARENT_RETRIES; ++parent_retry)
         {
            size_t parent_idx = parent_pool_indices[parent_dist(gen)];
            const auto &parent_labels = filtered_label_sets[parent_idx];

            for (int t = 0; t < TOURNAMENT_SIZE; ++t)
            {
               std::vector<LabelType> candidate_subset;
               std::vector<LabelType> sorted_parent = parent_labels;
               std::sort(sorted_parent.begin(), sorted_parent.end());

               size_t elite_pool_size = std::min(sorted_parent.size(), static_cast<size_t>(query_length * 3));
               std::shuffle(sorted_parent.begin(), sorted_parent.begin() + elite_pool_size, gen);

               candidate_subset.assign(sorted_parent.begin(), sorted_parent.begin() + query_length);
               std::sort(candidate_subset.begin(), candidate_subset.end());

               IdxType coverage = trie_index.calculate_coverage(candidate_subset);

               if (coverage < K || coverage > max_coverage)
               {
                  continue;
               }

               const TrieNode *node = trie_index.find_node(candidate_subset);
               IdxType children_count = (node) ? node->superset_count.load() : 0;

               if (children_count >= min_children)
               {
                  local_label_sets.push_back(candidate_subset);
                  local_vectors.push_back(filtered_vectors[parent_idx]);
                  query_generated = true;
                  break;
               }
            }

            if (query_generated)
            {
               break;
            }
         }
         // ===== NEW CODE START: Update and Report Progress =====
         size_t current_progress = ++progress_counter;
         if (current_progress % 50 == 0) // Print progress every 50 tasks
         {
#pragma omp critical
            {
               // A double-check to avoid printing stale info
               if (generated_label_sets.size() > last_reported_count || current_progress % 200 == 0)
               {
                  std::cout << "\r[Progress] Tasks completed: " << current_progress << "/" << num_points
                            << ". Queries generated: " << generated_label_sets.size() << "..." << std::flush;
                  last_reported_count = generated_label_sets.size();
               }
            }
         }
         // ===== NEW CODE END =====
      }

#pragma omp critical
      {
         generated_label_sets.insert(generated_label_sets.end(), local_label_sets.begin(), local_label_sets.end());
         generated_vectors.insert(generated_vectors.end(), local_vectors.begin(), local_vectors.end());
         // Final progress update upon thread completion
         std::cout << "\r[Progress] Tasks completed: " << progress_counter.load() << "/" << num_points
                   << ". Queries generated: " << generated_label_sets.size() << "..." << std::flush;
      }
   }

   size_t num_successfully_generated = generated_label_sets.size();
   if (num_successfully_generated < num_points && num_successfully_generated > 0)
   {
      std::cout << "\n[INFO] Generated " << num_successfully_generated << " unique queries. Starting smart padding to reach " << num_points << "..." << std::endl;
      size_t num_to_padd = num_points - num_successfully_generated;

      std::mt19937 gen(std::random_device{}());
      std::uniform_int_distribution<size_t> padding_labels_dist(0, num_successfully_generated - 1);
      std::uniform_int_distribution<size_t> padding_vectors_dist(0, all_base_vectors.size() - 1);

      for (size_t i = 0; i < num_to_padd; ++i)
      {
         size_t label_idx_to_copy = padding_labels_dist(gen);
         size_t vector_idx_to_sample = padding_vectors_dist(gen);

         generated_label_sets.push_back(generated_label_sets[label_idx_to_copy]);
         generated_vectors.push_back(all_base_vectors[vector_idx_to_sample]);
      }
      std::cout << "[INFO] Padded with " << num_to_padd << " queries (re-used labels, new random vectors)." << std::endl;
      std::cout << "[INFO] Total query count is now " << generated_label_sets.size() << "." << std::endl;
   }

   std::cout << "\nGeneration complete." << std::endl;
   if (generated_label_sets.size() < num_points)
   {
      std::cout << "Warning: Could only generate " << generated_label_sets.size() << " queries meeting the criteria, and padding failed because no queries were generated." << std::endl;
   }

   std::ofstream outfile(output_file);
   for (const auto &ls : generated_label_sets)
   {
      for (size_t k = 0; k < ls.size(); ++k)
      {
         outfile << ls[k] << (k == ls.size() - 1 ? "" : ",");
      }
      outfile << std::endl;
   }
   std::cout << generated_label_sets.size() << " query labels written to " << output_file << std::endl;

   write_fvecs(output_vectors_file, generated_vectors);
   std::cout << generated_vectors.size() << " query vectors written to " << output_vectors_file << std::endl;
}

void run_variable_sub_base_mode(const po::variables_map &vm)
{
   std::string input_file = vm["input_file"].as<std::string>();
   std::string output_file = vm["output_file"].as<std::string>();
   std::string base_vectors_file = vm["base_vectors_file"].as<std::string>();
   std::string output_vectors_file = vm["output_vectors_file"].as<std::string>();
   IdxType num_points = vm["num_points"].as<IdxType>();
   IdxType min_query_length = vm["min-query-length"].as<IdxType>();
   IdxType max_query_length = vm["max-query-length"].as<IdxType>();
   IdxType K = vm["K"].as<IdxType>();
   IdxType max_coverage = vm["max-coverage"].as<IdxType>();
   IdxType min_children = vm["min-children"].as<IdxType>();
   double parent_range_start = vm["parent-range-start"].as<double>();
   double parent_range_end = vm["parent-range-end"].as<double>();

   if (parent_range_start < 0.0 || parent_range_end > 1.0 || parent_range_start >= parent_range_end)
   {
      std::cerr << "Error: parent range must satisfy 0.0 <= start < end <= 1.0." << std::endl;
      return;
   }

   std::cout << "========================================================" << std::endl;
   std::cout << "--- Running in VARIABLE_SUB_BASE (Random Parent + Variable Subset) mode ---" << std::endl;
   std::cout << "Minimum query length: " << min_query_length << std::endl;
   std::cout << "Maximum query length: " << (max_query_length == 0 ? std::string("parent length") : std::to_string(max_query_length)) << std::endl;
   std::cout << "Target coverage range: [" << K << ", " << max_coverage << "]" << std::endl;
   std::cout << "Minimum children (supersets): " << min_children << std::endl;
   std::cout << "Parent label source range: [" << parent_range_start << ", " << parent_range_end << ")" << std::endl;
   std::cout << "========================================================" << std::endl;

   TrieIndex trie_index;
   std::ifstream infile(input_file);
   IdxType new_label_set_id = 1;
   std::string line, label_str;

   std::vector<std::vector<float>> all_base_vectors;
   int dim;
   try
   {
      load_fvecs(base_vectors_file, all_base_vectors, dim);
   }
   catch (const std::exception &e)
   {
      std::cerr << "Fatal Error loading base vectors: " << e.what() << std::endl;
      return;
   }

   std::vector<std::vector<LabelType>> filtered_label_sets;
   std::vector<std::vector<float>> filtered_vectors;
   std::vector<size_t> filtered_source_indices;

   size_t line_index = 0;
   while (std::getline(infile, line))
   {
      if (line_index >= all_base_vectors.size())
      {
         std::cerr << "Warning: Label file has more lines than vectors. Stopping at line " << line_index << std::endl;
         break;
      }

      std::vector<LabelType> label_set;
      std::stringstream ss(line);
      while (std::getline(ss, label_str, ','))
      {
         if (!label_str.empty())
         {
            label_set.emplace_back(std::stoul(label_str));
         }
      }

      if (!label_set.empty())
      {
         trie_index.insert(label_set, new_label_set_id);
         filtered_label_sets.push_back(label_set);
         filtered_vectors.push_back(all_base_vectors[line_index]);
         filtered_source_indices.push_back(line_index);
      }
      line_index++;
   }

   std::cout << "Successfully loaded and filtered " << filtered_label_sets.size()
             << " items with non-empty labels." << std::endl;

   std::cout << "Pre-computing subtree sums for coverage optimization..." << std::endl;
   trie_index.precompute_subtree_sums();
   std::cout << "Pre-computation complete." << std::endl;

   bool cache_loaded = false;
   if (vm.count("cache-file"))
   {
      std::string cache_path = vm["cache-file"].as<std::string>();
      if (trie_index.load_superset_cache(cache_path))
      {
         cache_loaded = true;
      }
   }

   if (!cache_loaded && min_children > 0)
   {
      trie_index.calculate_superset_counts();
      if (vm.count("cache-file"))
      {
         std::string cache_path = vm["cache-file"].as<std::string>();
         auto nodes = trie_index.get_all_nodes_with_group_id();
         trie_index.save_superset_cache(cache_path, nodes);
      }
   }

   const size_t base_label_count = std::min(line_index, all_base_vectors.size());
   const size_t parent_range_begin = static_cast<size_t>(std::floor(parent_range_start * static_cast<double>(base_label_count)));
   const size_t parent_range_end_exclusive = static_cast<size_t>(std::floor(parent_range_end * static_cast<double>(base_label_count)));

   std::vector<size_t> parent_pool_indices;
   for (size_t i = 0; i < filtered_label_sets.size(); ++i)
   {
      if (filtered_label_sets[i].size() < min_query_length)
      {
         continue;
      }

      const size_t source_index = filtered_source_indices[i];
      if (source_index >= parent_range_begin && source_index < parent_range_end_exclusive)
      {
         parent_pool_indices.push_back(i);
      }
   }

   if (parent_pool_indices.empty())
   {
      std::cerr << "Error: No base label sets found in parent range ["
                << parent_range_start << ", " << parent_range_end
                << ") with length >= " << min_query_length << ". Cannot generate queries." << std::endl;
      return;
   }

   std::cout << "Found " << parent_pool_indices.size()
             << " valid parent label sets for sampling in base index range ["
             << parent_range_begin << ", " << parent_range_end_exclusive << ")." << std::endl;

   std::vector<std::vector<LabelType>> slot_label_sets(num_points);
   std::vector<std::vector<float>> slot_vectors(num_points);
   std::vector<unsigned char> slot_generated(num_points, 0);
   std::atomic<size_t> generated_count{0};

   const int TOURNAMENT_SIZE = 20;
   const int MAX_PARENT_RETRIES = 500;

   std::atomic<size_t> progress_counter{0};
   size_t last_reported_count = 0;

#pragma omp parallel
   {
      std::mt19937 gen(std::random_device{}() + omp_get_thread_num());
      std::uniform_int_distribution<size_t> vector_dist(0, all_base_vectors.size() - 1);

#pragma omp for schedule(dynamic)
      for (IdxType i = 0; i < num_points; ++i)
      {
         std::uniform_int_distribution<size_t> parent_dist(0, parent_pool_indices.size() - 1);

         bool query_generated = false;
         for (int parent_retry = 0; parent_retry < MAX_PARENT_RETRIES; ++parent_retry)
         {
            size_t parent_idx = parent_pool_indices[parent_dist(gen)];
            const auto &parent_labels = filtered_label_sets[parent_idx];
            size_t effective_max_length = parent_labels.size();
            if (max_query_length > 0)
            {
               effective_max_length = std::min(effective_max_length, static_cast<size_t>(max_query_length));
            }
            if (effective_max_length < min_query_length)
            {
               continue;
            }
            std::uniform_int_distribution<size_t> length_dist(min_query_length, effective_max_length);

            for (int t = 0; t < TOURNAMENT_SIZE; ++t)
            {
               size_t query_length = length_dist(gen);
               std::vector<LabelType> candidate_subset = parent_labels;
               std::shuffle(candidate_subset.begin(), candidate_subset.end(), gen);
               candidate_subset.resize(query_length);
               std::sort(candidate_subset.begin(), candidate_subset.end());

               IdxType coverage = trie_index.calculate_coverage(candidate_subset);

               if (coverage < K || coverage > max_coverage)
               {
                  continue;
               }

               if (min_children > 0)
               {
                  const TrieNode *node = trie_index.find_node(candidate_subset);
                  IdxType children_count = (node) ? node->superset_count.load() : 0;
                  if (children_count < min_children)
                  {
                     continue;
                  }
               }

               slot_label_sets[i] = std::move(candidate_subset);
               slot_vectors[i] = all_base_vectors[vector_dist(gen)];
               slot_generated[i] = 1;
               ++generated_count;
               query_generated = true;
               break;
            }

            if (query_generated)
            {
               break;
            }
         }

         size_t current_progress = ++progress_counter;
         if (current_progress % 50 == 0)
         {
#pragma omp critical
            {
               size_t current_generated = generated_count.load();
               if (current_generated > last_reported_count || current_progress % 200 == 0)
               {
                  std::cout << "\r[Progress] Tasks completed: " << current_progress << "/" << num_points
                            << ". Queries generated: " << current_generated << "..." << std::flush;
                  last_reported_count = current_generated;
               }
            }
         }
      }
   }

   std::cout << "\r[Progress] Tasks completed: " << progress_counter.load() << "/" << num_points
             << ". Queries generated: " << generated_count.load() << "..." << std::flush;

   size_t num_successfully_generated = generated_count.load();
   if (num_successfully_generated < num_points && num_successfully_generated > 0)
   {
      std::cout << "\n[INFO] Generated " << num_successfully_generated << " unique queries. Starting smart padding to reach " << num_points << "..." << std::endl;

      std::vector<size_t> generated_slots;
      for (size_t i = 0; i < num_points; ++i)
      {
         if (slot_generated[i])
         {
            generated_slots.push_back(i);
         }
      }

      std::mt19937 gen(std::random_device{}());
      std::uniform_int_distribution<size_t> padding_vectors_dist(0, all_base_vectors.size() - 1);

      for (size_t i = 0; i < num_points; ++i)
      {
         if (slot_generated[i])
         {
            continue;
         }

         if (generated_slots.empty())
         {
            continue;
         }

         std::uniform_int_distribution<size_t> padding_labels_dist(0, generated_slots.size() - 1);
         size_t source_slot = generated_slots[padding_labels_dist(gen)];
         size_t vector_idx_to_sample = padding_vectors_dist(gen);

         slot_label_sets[i] = slot_label_sets[source_slot];
         slot_vectors[i] = all_base_vectors[vector_idx_to_sample];
         slot_generated[i] = 1;
         ++generated_count;
      }

      std::cout << "[INFO] Padded to " << generated_count.load() << " queries (re-used labels from the configured parent range, new random vectors)." << std::endl;
   }

   std::vector<std::vector<LabelType>> generated_label_sets;
   std::vector<std::vector<float>> generated_vectors;
   generated_label_sets.reserve(num_points);
   generated_vectors.reserve(num_points);
   for (size_t i = 0; i < num_points; ++i)
   {
      if (slot_generated[i])
      {
         generated_label_sets.push_back(std::move(slot_label_sets[i]));
         generated_vectors.push_back(std::move(slot_vectors[i]));
      }
   }

   std::cout << "\nGeneration complete." << std::endl;
   if (generated_label_sets.size() < num_points)
   {
      std::cout << "Warning: Could only generate " << generated_label_sets.size() << " queries meeting the criteria, and padding failed because no queries were generated." << std::endl;
   }

   std::ofstream outfile(output_file);
   for (const auto &ls : generated_label_sets)
   {
      for (size_t k = 0; k < ls.size(); ++k)
      {
         outfile << ls[k] << (k == ls.size() - 1 ? "" : ",");
      }
      outfile << std::endl;
   }
   std::cout << generated_label_sets.size() << " query labels written to " << output_file << std::endl;

   write_fvecs(output_vectors_file, generated_vectors);
   std::cout << generated_vectors.size() << " query vectors written to " << output_vectors_file << std::endl;
}


int main(int argc, char **argv)
{
   po::options_description generic_opts("Generic options");
   generic_opts.add_options()("help,h", "Print help message")("mode", boost::program_options::value<std::string>()->default_value("generate"), "Run mode: generate, analyze, sub_base, weighted_sub_base, variable_sub_base");

   po::options_description generate_opts("Mode 'generate': Handles 'uniform' and 'zipf' distributions");
   generate_opts.add_options()("input_file", boost::program_options::value<std::string>(), "Path to the base label file (required for all modes)")("output_file", boost::program_options::value<std::string>(), "Output path for the generated candidate labels")("base_vectors_file", boost::program_options::value<std::string>(), "[Optional] Path to the base vectors file (.fvecs format)")("output_vectors_file", boost::program_options::value<std::string>(), "[Optional] Output path for the generated query vectors (.fvecs format)")("num_points", boost::program_options::value<IdxType>()->default_value(10000), "Number of queries to generate")("K", boost::program_options::value<IdxType>()->default_value(20), "Minimum number of potential results for a query to be valid")("distribution_type", boost::program_options::value<std::string>()->default_value("uniform"), "Distribution type: uniform, zipf")("truncate_to_fixed_length", boost::program_options::value<bool>()->default_value(true), "Use fixed-length queries")("num_labels_per_query", boost::program_options::value<IdxType>()->default_value(2), "The fixed number of labels per query")("expected_num_label", boost::program_options::value<IdxType>()->default_value(3), "Expected number of labels per query (variable length)");

   po::options_description analyze_opts("Mode 'analyze': Analyze a candidate pool and profile coverage");
   analyze_opts.add_options()("candidate_file", boost::program_options::value<std::string>(), "Path to the candidate query file to be analyzed")("profiled_output", boost::program_options::value<std::string>(), "Output path for the analysis result (.csv)");

   po::options_description sub_base_opts("Modes sub_base/weighted_sub_base/variable_sub_base: Generate hard queries using subset expansion");
   sub_base_opts.add_options()("query-length", boost::program_options::value<IdxType>()->default_value(5), "Exact number of labels for fixed-length modes")("min-query-length", boost::program_options::value<IdxType>()->default_value(3), "Minimum number of labels for variable_sub_base")("max-query-length", boost::program_options::value<IdxType>()->default_value(0), "Maximum labels for variable_sub_base, 0 means parent length")("max-coverage", boost::program_options::value<IdxType>()->default_value(1000), "Maximum number of matching vectors for a valid query")("min-children", boost::program_options::value<IdxType>()->default_value(1), "Minimum number of supersets a query must have in the dataset")("parent-range-start", boost::program_options::value<double>()->default_value(0.0), "Start ratio of the base-label parent source range for variable_sub_base")("parent-range-end", boost::program_options::value<double>()->default_value(1.0), "End ratio of the base-label parent source range for variable_sub_base")("cache-file", boost::program_options::value<std::string>(), "[Optional] Path to save/load the pre-computation cache");

   po::options_description cmdline_opts;
   cmdline_opts.add(generic_opts).add(generate_opts).add(analyze_opts).add(sub_base_opts);

   po::variables_map vm;
   po::store(po::parse_command_line(argc, argv, cmdline_opts), vm);
   po::notify(vm);

   if (vm.count("help"))
   {
      std::cout << "Multi-purpose Query Generation and Analysis Tool." << std::endl
                << std::endl;
      std::cout << generic_opts << std::endl;
      std::cout << generate_opts << std::endl;
      std::cout << analyze_opts << std::endl;
      std::cout << sub_base_opts << std::endl;
      return 0;
   }

   std::string mode = vm["mode"].as<std::string>();
   if (mode == "generate" || mode == "analyze" || mode == "sub_base" || mode == "weighted_sub_base" || mode == "variable_sub_base")
   {
      if (!vm.count("input_file"))
      {
         std::cerr << "Error: Mode '" << mode << "' requires the --input_file parameter." << std::endl;
         return 1;
      }
   }

   if (mode == "generate")
   {
      std::cout << "Running in 'generate' mode..." << std::endl;
      run_generate_mode(vm);
   }
   else if (mode == "analyze")
   {
      std::cout << "Running in 'analyze' mode..." << std::endl;
      run_analyze_mode(vm);
   }
   else if (mode == "sub_base")
   {
      if (!vm.count("base_vectors_file") || !vm.count("output_vectors_file") || !vm.count("K"))
      {
         std::cerr << "Error: Mode 'sub_base' requires --base_vectors_file, --output_vectors_file, and --K." << std::endl;
         return 1;
      }
      run_sub_base_subset_mode(vm);
   }
   else if (mode == "weighted_sub_base")
   {
      if (!vm.count("base_vectors_file") || !vm.count("output_vectors_file") || !vm.count("K"))
      {
         std::cerr << "Error: Mode 'weighted_sub_base' requires --base_vectors_file, --output_vectors_file, and --K." << std::endl;
         return 1;
      }
      run_weighted_sub_base_mode(vm);
   }
   else if (mode == "variable_sub_base")
   {
      if (!vm.count("base_vectors_file") || !vm.count("output_vectors_file") || !vm.count("K"))
      {
         std::cerr << "Error: Mode 'variable_sub_base' requires --base_vectors_file, --output_vectors_file, and --K." << std::endl;
         return 1;
      }
      run_variable_sub_base_mode(vm);
   }
   else
   {
      std::cerr << "Error: Unknown mode '" << mode << "'. Valid modes are: generate, analyze, sub_base, weighted_sub_base, variable_sub_base" << std::endl;
      return 1;
   }

   return 0;
}