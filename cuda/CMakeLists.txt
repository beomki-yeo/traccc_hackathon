enable_language(CUDA)

find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

set( TRACCC_CUDA_DIR ${CMAKE_CURRENT_SOURCE_DIR} )
set( TRACCC_CUDA_ALGORITHMS_SOURCE_DIR ${TRACCC_CUDA_DIR}/src/algorithms )
set( TRACCC_CUDA_UTILS_SOURCE_DIR ${TRACCC_CUDA_DIR}/src/utils )

add_subdirectory(include/cuda/utils)
add_subdirectory(include/cuda/algorithms)

add_library(traccc_cuda INTERFACE)
target_link_libraries(traccc_cuda INTERFACE traccc::cuda::utils traccc::cuda::algorithms)
add_library(traccc::cuda ALIAS traccc_cuda)


