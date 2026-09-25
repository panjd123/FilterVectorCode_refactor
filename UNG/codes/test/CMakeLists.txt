# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_COMPILE_WARNING_AS_ERROR ON)

# add_executable(test_build_vamana test_build_vamana.cpp)
# target_link_libraries(test_build_vamana ${PROJECT_NAME} Vamana Boost::program_options ${ROARING_LIB})

# add_executable(test_search_vamana test_search_vamana.cpp)
# target_link_libraries(test_search_vamana ${PROJECT_NAME} Vamana Boost::program_options ${ROARING_LIB})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_COMPILE_WARNING_AS_ERROR ON)

add_executable(test_build_vamana test_build_vamana.cpp)
target_link_libraries(test_build_vamana PRIVATE -Wl,--whole-archive ${PROJECT_NAME} Vamana -Wl,--no-whole-archive Boost::program_options Boost::filesystem OpenMP::OpenMP_CXX ${ROARING_LIB} onnxruntime faiss)

add_executable(test_search_vamana test_search_vamana.cpp)
target_link_libraries(test_search_vamana PRIVATE -Wl,--whole-archive ${PROJECT_NAME} Vamana -Wl,--no-whole-archive Boost::program_options Boost::filesystem OpenMP::OpenMP_CXX ${ROARING_LIB} onnxruntime faiss)

add_executable(test_special_block_free_state test_special_block_free_state.cpp)
target_link_libraries(test_special_block_free_state PRIVATE -Wl,--whole-archive ${PROJECT_NAME} Vamana -Wl,--no-whole-archive Boost::program_options Boost::filesystem OpenMP::OpenMP_CXX ${ROARING_LIB} onnxruntime faiss)
add_test(NAME special_block_free_state COMMAND test_special_block_free_state)

add_executable(test_special_candidate_queue test_special_candidate_queue.cpp)
target_link_libraries(test_special_candidate_queue PRIVATE -Wl,--whole-archive ${PROJECT_NAME} Vamana -Wl,--no-whole-archive Boost::program_options Boost::filesystem OpenMP::OpenMP_CXX ${ROARING_LIB} onnxruntime faiss)
add_test(NAME special_candidate_queue COMMAND test_special_candidate_queue)

add_executable(test_lng_block_partition test_lng_block_partition.cpp)
target_link_libraries(test_lng_block_partition PRIVATE -Wl,--whole-archive ${PROJECT_NAME} Vamana -Wl,--no-whole-archive Boost::program_options Boost::filesystem OpenMP::OpenMP_CXX ${ROARING_LIB} onnxruntime faiss)
add_test(NAME lng_block_partition COMMAND test_lng_block_partition)

add_executable(test_favor_block_search test_favor_block_search.cpp)
target_link_libraries(test_favor_block_search PRIVATE -Wl,--whole-archive ${PROJECT_NAME} Vamana -Wl,--no-whole-archive Boost::program_options Boost::filesystem OpenMP::OpenMP_CXX ${ROARING_LIB} onnxruntime faiss)
add_test(NAME favor_block_search COMMAND test_favor_block_search)

add_executable(test_cpu_bruteforce_els test_cpu_bruteforce_els.cpp)
target_link_libraries(test_cpu_bruteforce_els PRIVATE -Wl,--whole-archive ${PROJECT_NAME} Vamana -Wl,--no-whole-archive Boost::program_options Boost::filesystem OpenMP::OpenMP_CXX ${ROARING_LIB} onnxruntime faiss)
add_test(NAME cpu_bruteforce_els COMMAND test_cpu_bruteforce_els)
add_executable(test_entry_group_cache test_entry_group_cache.cpp)
target_link_libraries(test_entry_group_cache PRIVATE -Wl,--whole-archive ${PROJECT_NAME} Vamana -Wl,--no-whole-archive Boost::program_options Boost::filesystem OpenMP::OpenMP_CXX ${ROARING_LIB} onnxruntime faiss)
add_test(NAME entry_group_cache COMMAND test_entry_group_cache)

add_executable(test_search_execution_context test_search_execution_context.cpp)
target_link_libraries(test_search_execution_context PRIVATE -Wl,--whole-archive ${PROJECT_NAME} Vamana -Wl,--no-whole-archive Boost::program_options Boost::filesystem OpenMP::OpenMP_CXX ${ROARING_LIB} onnxruntime faiss)
add_test(NAME search_execution_context COMMAND test_search_execution_context)


add_executable(test_navix_hnsw test_navix_hnsw.cpp)
target_link_libraries(test_navix_hnsw PRIVATE -Wl,--whole-archive ${PROJECT_NAME} Vamana -Wl,--no-whole-archive Boost::program_options Boost::filesystem OpenMP::OpenMP_CXX ${ROARING_LIB} onnxruntime faiss)
add_test(NAME navix_hnsw COMMAND test_navix_hnsw)
add_executable(test_special_edge_io test_special_edge_io.cpp)
target_link_libraries(test_special_edge_io PRIVATE -Wl,--whole-archive ${PROJECT_NAME} Vamana -Wl,--no-whole-archive Boost::program_options Boost::filesystem OpenMP::OpenMP_CXX ${ROARING_LIB} onnxruntime faiss)
add_test(NAME special_edge_io COMMAND test_special_edge_io)

add_executable(test_ung_build_config test_ung_build_config.cpp)
target_link_libraries(test_ung_build_config PRIVATE -Wl,--whole-archive ${PROJECT_NAME} Vamana -Wl,--no-whole-archive Boost::program_options Boost::filesystem OpenMP::OpenMP_CXX ${ROARING_LIB} onnxruntime faiss)
add_test(NAME ung_build_config COMMAND test_ung_build_config)

add_executable(test_ung_group_topology test_ung_group_topology.cpp)
target_link_libraries(test_ung_group_topology PRIVATE -Wl,--whole-archive ${PROJECT_NAME} Vamana -Wl,--no-whole-archive Boost::program_options Boost::filesystem OpenMP::OpenMP_CXX ${ROARING_LIB} onnxruntime faiss)
add_test(NAME ung_group_topology COMMAND test_ung_group_topology)

add_executable(test_special_block_trie test_special_block_trie.cpp)
target_link_libraries(test_special_block_trie PRIVATE -Wl,--whole-archive ${PROJECT_NAME} Vamana -Wl,--no-whole-archive Boost::program_options Boost::filesystem OpenMP::OpenMP_CXX ${ROARING_LIB} onnxruntime faiss)
add_test(NAME special_block_trie COMMAND test_special_block_trie)

add_executable(test_filter_validation test_filter_validation.cpp)
target_link_libraries(test_filter_validation PRIVATE -Wl,--whole-archive ${PROJECT_NAME} Vamana -Wl,--no-whole-archive Boost::program_options Boost::filesystem OpenMP::OpenMP_CXX ${ROARING_LIB} onnxruntime faiss)
add_test(NAME filter_validation COMMAND test_filter_validation)


add_executable(test_navix_search test_navix_search.cpp)
target_link_libraries(test_navix_search PRIVATE -Wl,--whole-archive ${PROJECT_NAME} Vamana -Wl,--no-whole-archive Boost::program_options Boost::filesystem OpenMP::OpenMP_CXX ${ROARING_LIB} onnxruntime faiss)
add_test(NAME navix_search COMMAND test_navix_search)

add_executable(test_navix_index test_navix_index.cpp)
target_link_libraries(test_navix_index PRIVATE -Wl,--whole-archive ${PROJECT_NAME} Vamana -Wl,--no-whole-archive Boost::program_options Boost::filesystem OpenMP::OpenMP_CXX ${ROARING_LIB} onnxruntime faiss)
add_test(NAME navix_index COMMAND test_navix_index)
