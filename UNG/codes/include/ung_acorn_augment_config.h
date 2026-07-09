#ifndef UNG_ACORN_AUGMENT_CONFIG_H
#define UNG_ACORN_AUGMENT_CONFIG_H

#include <string>

namespace ANNS
{
   // Legacy ACORN-in-UNG augmentation settings. This path invokes an external
   // ACORN script and is retained for historical experiments, not for the
   // current GPU cross-edge main path.
   struct AcornInUng
   {
      bool ung_and_acorn = false;
      std::string new_edge_policy = "false"; // false / just_acorn / just_guarantee / acorn_and_guarantee
      int R_in_add_new_edge = 50;    // ACORN neighbors requested per sampled query vector
      int W_in_add_new_edge = 50;    // Historical ACORN pruning width; currently passed through for compatibility
      int M_in_add_new_edge = 4;     // Maximum new in/out degree for distance-oriented edges
      float layer_depth_retio = 0.8f; // Historical misspelling kept for CLI/config compatibility
      float query_vector_ratio = 0.8f; // Sampling ratio for candidate query vectors
      float root_coverage_threshold = 0.4f; // Minimum label coverage to be treated as a conceptual root
      std::string acorn_in_ung_output_path = "false";

      // ACORN script parameters.
      int M = 32;
      int M_beta = 64;
      int gamma = 80;
      int efs = 1000;
      int compute_recall = 1;
   };
}

#endif // UNG_ACORN_AUGMENT_CONFIG_H
