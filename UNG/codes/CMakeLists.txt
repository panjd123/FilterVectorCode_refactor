cmake_minimum_required(VERSION 3.15)
project(ANNS)
enable_testing()

# Keep external dependencies relocatable. By default this repository is assumed
# to live directly under the graphdb workspace root, but callers can override
# either path when using a different layout.
get_filename_component(_DEFAULT_GRAPHDB_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
set(GRAPHDB_ROOT "${_DEFAULT_GRAPHDB_ROOT}" CACHE PATH "graphdb workspace root")
set(TAGORE_ROOT "${GRAPHDB_ROOT}/Tagore" CACHE PATH "Tagore source tree")
set(BOOST_ROOT "${GRAPHDB_ROOT}/boost_1_89_0" CACHE PATH "Boost installation prefix")
set(Boost_ROOT "${BOOST_ROOT}")
option(UNG_BOOST_USE_CXX11_ABI "Use the new libstdc++ C++11 ABI for the selected Boost binaries" OFF)
if(NOT UNG_BOOST_USE_CXX11_ABI AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # The bundled boost_1_89_0 binaries in this workspace were built with the
    # legacy libstdc++ ABI (their exported symbols use std::string, not
    # std::__cxx11::basic_string).  Keep all targets ABI-consistent.
    add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
endif()
set(OPENBLAS_ROOT "${GRAPHDB_ROOT}/openblas" CACHE PATH "OpenBLAS installation prefix")
set(ZLIB_ROOT "${GRAPHDB_ROOT}/zlib" CACHE PATH "zlib installation prefix")
list(PREPEND CMAKE_PREFIX_PATH "${OPENBLAS_ROOT}" "${ZLIB_ROOT}")

# Make locally bundled/shared dependencies discoverable when running binaries
# straight from the build tree.
list(APPEND CMAKE_BUILD_RPATH
    "${BOOST_ROOT}/lib"
    "${OPENBLAS_ROOT}/lib"
    "${ZLIB_ROOT}/lib"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime-linux-x64-1.16.3/lib")

option(UNG_ENABLE_CUDA "Build CUDA kernels and GPU search paths" ON)
if(UNG_ENABLE_CUDA)
    # Enable CUDA for GPU cross-group edge kernels.
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES 86)
        endif()
    else()
        message(WARNING "CUDA compiler not found; configuring CPU sources only. Set UNG_ENABLE_CUDA=ON on a CUDA toolchain.")
        set(UNG_ENABLE_CUDA OFF CACHE BOOL "Build CUDA kernels and GPU search paths" FORCE)
    endif()
endif()

# cmake setting
set(CMAKE_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 将 ACORN/faiss 项目作为 UNG 项目的一个子目录添加进来,这样 UNG 就能找到 faiss 定义的所有目标 (targets);第二个参数指定了构建输出目录，避免污染 faiss 的源文件目录
set(FAISS_ENABLE_GPU OFF CACHE BOOL "Disable Faiss GPU support")
set(FAISS_ENABLE_PYTHON OFF CACHE BOOL "Disable Faiss Python bindings")
set(BUILD_TESTING OFF CACHE BOOL "Disable Faiss tests")
add_subdirectory(../../ACORN ${CMAKE_BINARY_DIR}/acorn_build)

# Boost
# set(Boost_INCLUDEDIRS "specify if needed")
# set(Boost_ROOT "specify if needed")
# set(Boost_DIR "specify if needed")
find_package(Boost COMPONENTS REQUIRED program_options filesystem)
message(STATUS "Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")
if (Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIR})
else()
    message(FATAL_ERROR "no Boost supprot")
endif()

# OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(FATAL_ERROR "no OpenMP supprot")
endif()

# build options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma -msse2 -ftree-vectorize -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free -fopenmp -fopenmp-simd -funroll-loops -Wfatal-errors -DUSE_AVX2")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -g -DNDEBUG -Ofast -march=native -mtune=native")

# Find pre-built CRoaring library
set(CROARING_ROOT "${PROJECT_SOURCE_DIR}/third_party/CRoaring" CACHE PATH "CRoaring source/install root")
set(ROARING_LIB_DIR "${CROARING_ROOT}/build/src")
find_library(ROARING_LIB
    NAMES roaring libroaring
    PATHS ${ROARING_LIB_DIR} "${CROARING_ROOT}/install/lib"
    PATH_SUFFIXES lib lib64
    NO_DEFAULT_PATH
)
if(NOT ROARING_LIB)
    message(WARNING "CRoaring prebuilt library not found in ${ROARING_LIB_DIR}. Falling back to building bundled CRoaring.")
    add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/CRoaring ${CMAKE_BINARY_DIR}/croaring_build)
    set(ROARING_LIB roaring)
else()
    message(STATUS "CRoaring library found at: ${ROARING_LIB}")
endif()

# ONNX
set(ONNXRUNTIME_DIR "${PROJECT_SOURCE_DIR}/third_party/onnxruntime-linux-x64-1.16.3" CACHE PATH "Path to ONNX Runtime directory")
if(EXISTS ${ONNXRUNTIME_DIR})
    include_directories(${ONNXRUNTIME_DIR}/include)
    add_library(onnxruntime SHARED IMPORTED)
    set_target_properties(onnxruntime PROPERTIES
        IMPORTED_LOCATION "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so"
    )
    message(STATUS "Found ONNX Runtime at: ${ONNXRUNTIME_DIR}")
else()
    message(FATAL_ERROR "ONNX Runtime not found at specified path: ${ONNXRUNTIME_DIR}. Please check the path in CMakeLists.txt")
endif()

# add subdirectories
add_subdirectory(vamana)
add_subdirectory(src)
option(UNG_BUILD_TOOLS "Build command-line tools" ${UNG_ENABLE_CUDA})
option(UNG_BUILD_TESTS "Build test executables" ${UNG_ENABLE_CUDA})
option(UNG_BUILD_APPS "Build index/search applications" ${UNG_ENABLE_CUDA})
if(UNG_BUILD_TOOLS)
    add_subdirectory(tools)
endif()
if(UNG_BUILD_TESTS)
    add_subdirectory(test)
endif()
if(UNG_BUILD_APPS)
    add_subdirectory(apps)
endif()
