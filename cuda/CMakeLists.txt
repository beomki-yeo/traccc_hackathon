enable_language(CUDA)

add_library(traccc_cuda SHARED
  ${CMAKE_CURRENT_SOURCE_DIR}/src/algorithms/seeding/doublet_counting.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/src/algorithms/seeding/doublet_finding.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/src/algorithms/seeding/triplet_finding.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/src/algorithms/seeding/triplet_counting.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/src/algorithms/seeding/weight_updating.cu)

find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

target_include_directories(traccc_cuda
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  )

target_compile_options(traccc_cuda
  PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
  --expt-relaxed-constexpr
  --fmad=false>)

target_link_libraries(traccc_cuda PUBLIC traccc::core vecmem::core vecmem::cuda)

install(
  DIRECTORY include/cuda/algorithms
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(
  DIRECTORY include/cuda/utils
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

add_library(traccc::cuda ALIAS traccc_cuda)
