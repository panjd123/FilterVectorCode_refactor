cmake_minimum_required(VERSION 3.17)
project(pathseer_project_adapter LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PATHSEER_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." CACHE PATH "PathSeer source root")
set(PATHSEER_FAISS_BUILD_DIR "${PATHSEER_ROOT}/faiss/build" CACHE PATH
    "PathSeer FAISS build directory")

find_package(OpenMP REQUIRED)

add_library(pathseer_faiss SHARED IMPORTED GLOBAL)
set_target_properties(pathseer_faiss PROPERTIES
    IMPORTED_LOCATION "${PATHSEER_FAISS_BUILD_DIR}/faiss/libfaiss.so"
    INTERFACE_INCLUDE_DIRECTORIES "${PATHSEER_ROOT}/faiss"
)

function(add_pathseer_adapter target source output_name)
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE pathseer_faiss OpenMP::OpenMP_CXX)
    target_compile_options(${target} PRIVATE -O3 -march=native)
    set_target_properties(${target} PROPERTIES
        OUTPUT_NAME "${output_name}"
        BUILD_RPATH "${PATHSEER_FAISS_BUILD_DIR}/faiss"
    )
endfunction()

add_pathseer_adapter(pathseer_build_index build_index.cpp build_index)
add_pathseer_adapter(pathseer_search_sweep search_sweep.cpp search_sweep)
