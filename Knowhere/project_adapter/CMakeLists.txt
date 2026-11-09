cmake_minimum_required(VERSION 3.17)
project(milvus_hnsw_project_adapter LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(KNOWHERE_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." CACHE PATH "Knowhere source root")
set(KNOWHERE_FAISS_BUILD_DIR "${KNOWHERE_ROOT}/thirdparty/faiss/build_adapter" CACHE PATH
    "Build directory for Knowhere's bundled FAISS")

find_package(OpenMP REQUIRED)
find_library(OPENBLAS_LIBRARY NAMES openblas PATHS /home/dev/graphdb/openblas/lib REQUIRED)

add_library(knowhere_faiss SHARED IMPORTED GLOBAL)
set_target_properties(knowhere_faiss PROPERTIES
    IMPORTED_LOCATION "${KNOWHERE_FAISS_BUILD_DIR}/faiss/libfaiss.so"
    INTERFACE_INCLUDE_DIRECTORIES "${KNOWHERE_ROOT}/thirdparty/faiss"
)

add_library(milvus_hnsw_core STATIC
    "${KNOWHERE_ROOT}/thirdparty/faiss/faiss/cppcontrib/knowhere/IndexFlat.cpp"
    "${KNOWHERE_ROOT}/thirdparty/faiss/faiss/cppcontrib/knowhere/IndexHNSW.cpp"
    "${KNOWHERE_ROOT}/thirdparty/faiss/faiss/cppcontrib/knowhere/impl/HNSW.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/jaccard_stub.cpp"
)
target_include_directories(milvus_hnsw_core PUBLIC
    "${CMAKE_CURRENT_LIST_DIR}/stubs"
    "${KNOWHERE_ROOT}"
    "${KNOWHERE_ROOT}/src"
    "${KNOWHERE_ROOT}/include"
    "${KNOWHERE_ROOT}/thirdparty/faiss"
)
target_link_libraries(milvus_hnsw_core PUBLIC knowhere_faiss OpenMP::OpenMP_CXX ${OPENBLAS_LIBRARY})
target_compile_definitions(milvus_hnsw_core PRIVATE FINTEGER=int)
target_compile_options(milvus_hnsw_core PRIVATE -O3 -march=native)

function(add_milvus_hnsw_adapter target source output_name)
    add_executable(${target} ${source})
    target_include_directories(${target} PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}/stubs"
        "${KNOWHERE_ROOT}"
        "${KNOWHERE_ROOT}/src"
        "${KNOWHERE_ROOT}/include"
        "${KNOWHERE_ROOT}/thirdparty/faiss"
    )
    target_link_libraries(${target} PRIVATE milvus_hnsw_core knowhere_faiss OpenMP::OpenMP_CXX)
    target_compile_options(${target} PRIVATE -O3 -march=native)
    set_target_properties(${target} PROPERTIES
        OUTPUT_NAME "${output_name}"
        BUILD_RPATH "${KNOWHERE_FAISS_BUILD_DIR}/faiss"
    )
endfunction()

add_milvus_hnsw_adapter(milvus_hnsw_build_index build_index.cpp build_index)
add_milvus_hnsw_adapter(milvus_hnsw_search_sweep search_sweep.cpp search_sweep)
