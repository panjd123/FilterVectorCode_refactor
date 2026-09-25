#include "ung_build_config.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace
{
void expect(bool condition, const char *message)
{
   if (!condition)
   {
      std::cerr << "FAILED: " << message << '\n';
      std::exit(1);
   }
}

bool overridden_thresholds_throw(uint32_t min_points)
{
   try
   {
      (void)ANNS::UngBuildConfig::from_env(1, min_points);
   }
   catch (const std::invalid_argument &)
   {
      return true;
   }
   return false;
}
} // namespace

int main()
{
   setenv("UNG_SPECIAL_BLOCK_MIN_POINTS", "50000", 1);
   setenv("UNG_SPECIAL_BLOCK_UPPER_MIN_POINTS", "10000", 1);
   const ANNS::UngBuildConfig cli_override =
       ANNS::UngBuildConfig::from_env(1, 1000);
   expect(cli_override.special_block_min_points == 1000 &&
              cli_override.special_block_upper_min_points == 10000,
          "CLI T1 must be the owner used to validate environment T2");

   expect(overridden_thresholds_throw(10000),
          "T2 equal to the final CLI T1 must be rejected");
   expect(overridden_thresholds_throw(20000),
          "T2 below the final CLI T1 must be rejected");
   expect(overridden_thresholds_throw(0),
          "zero CLI T1 must be rejected");

   unsetenv("UNG_SPECIAL_BLOCK_MIN_POINTS");
   unsetenv("UNG_SPECIAL_BLOCK_UPPER_MIN_POINTS");
   setenv("UNG_SPECIAL_BLOCKS", "1", 1);
   setenv("UNG_BASE_GROUP_TOPOLOGY", "trie", 1);
   setenv("UNG_HIERARCHY_LAYERS", "1000:trie,16000:lng,400000:trie", 1);
   const ANNS::UngBuildConfig multilevel = ANNS::UngBuildConfig::from_env(1);
   expect(multilevel.special_blocks_enabled &&
              multilevel.hierarchy.base_topology == ANNS::GroupTopologyKind::Trie &&
              multilevel.hierarchy.layer_count() == 3 &&
              multilevel.hierarchy.layers[0].min_points == 1000 &&
              multilevel.hierarchy.layers[1].topology == ANNS::GroupTopologyKind::Lng &&
              multilevel.hierarchy.layers[2].min_points == 400000,
          "build config must preserve an arbitrary mixed-topology hierarchy plan");

   setenv("UNG_SPECIAL_BLOCKS", "0", 1);
   unsetenv("UNG_HIERARCHY_LAYERS");
   const ANNS::UngBuildConfig zero_layer = ANNS::UngBuildConfig::from_env(1);
   expect(zero_layer.hierarchy.empty(),
          "disabling special blocks must produce a zero-layer plan");

   unsetenv("UNG_SPECIAL_BLOCKS");
   unsetenv("UNG_BASE_GROUP_TOPOLOGY");
   std::cout << "UNG build config checks passed\n";
   return 0;
}
