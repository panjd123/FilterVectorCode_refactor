set(CMAKE_CXX_STANDARD 17)
set(CMAKE_COMPILE_WARNING_AS_ERROR ON)

set(CPP_SOURCES vamana.cpp)
add_library(Vamana ${CPP_SOURCES})
target_include_directories(Vamana PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/third_party/CRoaring/include)

target_link_libraries(Vamana ${PROJECT_NAME} Boost::filesystem ${ROARING_LIB} onnxruntime faiss)
