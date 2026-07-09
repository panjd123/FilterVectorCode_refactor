#ifndef TRIE_TREE_H
#define TRIE_TREE_H

#include <vector>
#include <map>
#include <memory>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include "config.h"

namespace ANNS
{

   // Metrics collected by the shortcut-style trie entrance search.
   struct TrieMethod1Metrics
   {
      size_t initial_candidates = 0;
      size_t successful_checks = 0;

      long long upward_traversals = 0;
      long long bfs_nodes_processed = 0;

      long long redundant_upward_steps = 0;
   };

   // Metrics collected by the recursive trie entrance search.
   struct TrieSearchMetricsRecursive
   {
      long long recursive_calls = 0;
      long long pruning_events = 0;
      int max_recursion_depth = 0;

      long long collection_calls = 0;
      long long nodes_processed_in_bfs = 0;
      double time_in_collection_bfs = 0.0;
   };

   // Static trie structure summary used by query-route selectors.
   struct TrieStaticMetrics
   {
      size_t label_cardinality = 0;
      size_t total_nodes = 0;
      float avg_path_length = 0.0;
      float avg_branching_factor = 0.0;
      std::map<ANNS::LabelType, size_t> label_frequency;
   };

   // trie tree node
   struct TrieNode
   {
      LabelType label;
      IdxType group_id;         // group_id>0, and 0 if not a terminal node
      LabelType label_set_size; // number of elements in the label set if it is a terminal node
      IdxType group_size;       // number of elements in the group if it is a terminal node

      std::shared_ptr<TrieNode> parent;
      // std::map<LabelType, std::shared_ptr<TrieNode>> children;
      std::unordered_map<LabelType, std::shared_ptr<TrieNode>> children;

      TrieNode(LabelType x, std::shared_ptr<TrieNode> y)
          : label(x), parent(y), group_id(0), label_set_size(0), group_size(0) {}
      TrieNode(LabelType a, IdxType b, LabelType c, IdxType d)
          : label(a), group_id(b), label_set_size(c), group_size(d) {}
      ~TrieNode() = default;
   };

   // trie tree construction and search for super sets
   class TrieIndex
   {

   public:
      TrieIndex();

      // construction
      IdxType insert(const std::vector<LabelType> &label_set, IdxType &new_label_set_id);

      // query
      LabelType get_max_label_id() const { return _max_label_id; }
      std::shared_ptr<TrieNode> find_exact_match(const std::vector<LabelType> &label_set) const;
      void get_super_set_entrances(const std::vector<LabelType> &label_set,
                                   std::vector<std::shared_ptr<TrieNode>> &super_set_entrances,
                                   bool avoid_self = false, bool need_containment = true) const;
      void get_super_set_entrances_debug(const std::vector<LabelType> &label_set,
                                         std::vector<std::shared_ptr<TrieNode>> &super_set_entrances,
                                         bool avoid_self, bool need_containment,
                                         std::atomic<int> &print_counter, TrieMethod1Metrics &metrics,
                                         bool skip_group_id_check) const;
      void get_super_set_entrances_new_debug(const std::vector<LabelType> &label_set,
                                             std::vector<std::shared_ptr<TrieNode>> &super_set_entrances,
                                             bool avoid_self, bool need_containment,
                                             std::atomic<int> &print_counter, TrieSearchMetricsRecursive &metrics) const;
      void get_super_set_entrances_new_more_sp_debug(const std::vector<LabelType> &label_set,
                                                     std::vector<std::shared_ptr<TrieNode>> &super_set_entrances,
                                                     bool avoid_self, bool need_containment,
                                                     std::atomic<int> &print_counter, TrieSearchMetricsRecursive &metrics) const;

      size_t get_candidate_count_for_label(LabelType label) const
      {
         if (label >= _label_to_nodes.size())
         {
            return 0;
         }
         return _label_to_nodes[label].size();
      }
      TrieStaticMetrics calculate_static_metrics() const;

      // I/O
      void save(std::string filename) const;
      void load(std::string filename);
      float get_index_size();

   private:
      LabelType _max_label_id = 0;
      std::shared_ptr<TrieNode> _root;
      std::vector<std::vector<std::shared_ptr<TrieNode>>> _label_to_nodes;

      // Helpers for super-set lookup and selector-feature instrumentation.
      bool examine_smallest(const std::vector<LabelType> &label_set, const std::shared_ptr<TrieNode> &node) const;
      bool examine_containment(const std::vector<LabelType> &label_set, const std::shared_ptr<TrieNode> &node) const;
      bool examine_containment_debug(const std::vector<LabelType> &label_set,
                                     const std::shared_ptr<TrieNode> &node,
                                     long long &nodes_traversed,
                                     std::unordered_set<std::shared_ptr<TrieNode>> &visited_upward,
                                     long long &redundant_steps) const;
      void find_supersets_recursive_debug(
          std::shared_ptr<TrieNode> current_node,
          const std::vector<LabelType> &sorted_query,
          size_t query_idx,
          std::vector<std::shared_ptr<TrieNode>> &results,
          std::set<IdxType> &visited_groups,
          const std::shared_ptr<TrieNode> &avoided_node,
          TrieSearchMetricsRecursive &metrics,
          int current_depth) const;

      void find_supersets_iterative_debug(
          std::shared_ptr<TrieNode> start_node,
          const std::vector<LabelType> &sorted_query,
          size_t start_query_idx,
          std::vector<std::shared_ptr<TrieNode>> &results,
          std::set<IdxType> &visited_groups,
          const std::shared_ptr<TrieNode> &avoided_node,
          TrieSearchMetricsRecursive &metrics) const;

      void collect_all_terminals_debug(
          std::shared_ptr<TrieNode> start_node,
          std::vector<std::shared_ptr<TrieNode>> &results,
          std::set<IdxType> &visited_groups,
          const std::shared_ptr<TrieNode> &avoided_node,
          TrieSearchMetricsRecursive &metrics) const;
   };
}

#endif // TRIE_TREE_H
