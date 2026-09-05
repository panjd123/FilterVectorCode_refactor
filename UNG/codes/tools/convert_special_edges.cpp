#include "ung_special_edge_io.h"

#include <iostream>
#include <string>

int main(int argc, char **argv)
{
   if (argc != 3)
   {
      std::cerr << "Usage: convert_special_edges <special_edges.csv> <special_edges.bin>\n";
      return 2;
   }

   uint64_t count = 0;
   std::string error;
   if (!ANNS::convert_special_edge_csv_to_binary(argv[1], argv[2], count, error))
   {
      std::cerr << "special-edge conversion failed: " << error << '\n';
      return 1;
   }
   std::cout << "Converted " << count << " special edges to " << argv[2] << '\n';
   return 0;
}
