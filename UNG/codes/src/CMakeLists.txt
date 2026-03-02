set(CMAKE_CXX_STANDARD 17)
set(CMAKE_COMPILE_WARNING_AS_ERROR ON)

set(CPP_SOURCES utils.cpp storage.cpp trie.cpp distance.cpp search_queue.cpp filtered_scan.cpp uni_nav_graph.cpp MethodSelector.cpp)

set(CUDA_SOURCES
    gpu_gemm_topk.cu
)


# add_library(${PROJECT_NAME} ${CPP_SOURCES} ${ROARING_LIB})
add_library(${PROJECT_NAME} ${CPP_SOURCES} ${CUDA_SOURCES})

# 让这个库同时具备 C++/CUDA 的链接能力
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(${PROJECT_NAME} PRIVATE faiss ${ROARING_LIB} cuda cudart cublas cublasLt)

# 给 CUDA 源文件单独设置编译选项（避免吃到 CXX 的 avx/openmp flags）
set_source_files_properties(gpu_gemm_topk.cu PROPERTIES
    LANGUAGE CUDA
)

target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)
