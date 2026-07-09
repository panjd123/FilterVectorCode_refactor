set(CMAKE_CXX_STANDARD 17)
set(CMAKE_COMPILE_WARNING_AS_ERROR ON)

set(CPP_SOURCES
    utils.cpp
    storage.cpp
    trie.cpp
    distance.cpp
    search_queue.cpp
    filtered_scan.cpp

    # UNG orchestration and build-time modules
    uni_nav_graph.cpp
    uni_nav_graph_label_graph.cpp
    uni_nav_graph_group_graph.cpp
    uni_nav_graph_cross_edges.cpp
    uni_nav_graph_vector_attr.cpp
    uni_nav_graph_acorn_augment.cpp

    # UNG query/search and persistence
    uni_nav_graph_search.cpp
    uni_nav_graph_search_backend.cpp
    uni_nav_graph_query_route.cpp
    uni_nav_graph_query_features.cpp
    uni_nav_graph_entry_provider.cpp
    uni_nav_graph_special_blocks.cpp
    ung_graph_search_backend.cpp
    uni_nav_graph_io.cpp

    # Runtime method selection/configuration
    MethodSelector.cpp
    ung_build_config.cpp
    ung_build_settings.cpp
    ung_cross_edge_config.cpp
    ung_cross_edge_output_writer.cpp
    ung_group_graph_build_types.cpp
    tagore_cuda_config.cpp
    ung_entry_group.cpp
    ung_prof_log.cpp
)

set(CUDA_SOURCES
    gpu_gemm_topk.cu
    ung_gpu_cover_frontier_provider.cu
    tagore_graph_builder.cu
    /home/graphdb/Tagore/src/Tagore_src.cu
)


# add_library(${PROJECT_NAME} ${CPP_SOURCES} ${ROARING_LIB})
add_library(${PROJECT_NAME} ${CPP_SOURCES} ${CUDA_SOURCES})

# 让这个库同时具备 C++/CUDA 的链接能力
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

option(UNG_ENABLE_CUVS "Enable cuVS brute-force cross-edge baseline" OFF)
if(UNG_ENABLE_CUVS)
    list(APPEND CMAKE_PREFIX_PATH "/home/mabojing/opt/anaconda3")
    find_package(cuvs REQUIRED CONFIG)
    target_compile_definitions(${PROJECT_NAME} PRIVATE ANNS_HAVE_CUVS=1)
    target_link_libraries(${PROJECT_NAME} PRIVATE cuvs::cuvs)
endif()

target_link_libraries(${PROJECT_NAME} PRIVATE faiss ${ROARING_LIB} cuda cudart cublas cublasLt)

# 给 CUDA 源文件单独设置编译选项（避免吃到 CXX 的 avx/openmp flags）
set_source_files_properties(gpu_gemm_topk.cu ung_gpu_cover_frontier_provider.cu tagore_graph_builder.cu /home/graphdb/Tagore/src/Tagore_src.cu PROPERTIES
    LANGUAGE CUDA
)
set_source_files_properties(/home/graphdb/Tagore/src/Tagore_src.cu PROPERTIES
    COMPILE_OPTIONS "-Xcudafe=--diag_suppress=780;-Xcudafe=--diag_suppress=177;-Xcudafe=--diag_suppress=550"
)

target_include_directories(${PROJECT_NAME} PRIVATE /home/graphdb/Tagore/include)

target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)
