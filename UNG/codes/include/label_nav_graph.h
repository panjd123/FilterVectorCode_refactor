#ifndef LABEL_NAV_GRAPH_H
#define LABEL_NAV_GRAPH_H

#include <vector>
#include "config.h"

namespace ANNS
{

   class LabelNavGraph
   {

   public:
      LabelNavGraph(IdxType num_nodes)
      {
         in_neighbors.resize(num_nodes + 1);
         out_neighbors.resize(num_nodes + 1);
         in_degree.resize(num_nodes + 1, 0);
         out_degree.resize(num_nodes + 1, 0);

      };

      std::vector<std::vector<IdxType>> in_neighbors, out_neighbors;
      std::vector<int> in_degree, out_degree;            // 入度和出度

      ~LabelNavGraph() = default;

   private:
   };
}

#endif // LABEL_NAV_GRAPH_H
