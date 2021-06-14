add_library(traccc_cuda_algorithms SHARED
  ${TRACCC_CUDA_ALGORITHMS_SOURCE_DIR}/seeding/doublet_counting.cu
  ${TRACCC_CUDA_ALGORITHMS_SOURCE_DIR}/seeding/doublet_finding.cu
  ${TRACCC_CUDA_ALGORITHMS_SOURCE_DIR}/seeding/triplet_counting.cu
  ${TRACCC_CUDA_ALGORITHMS_SOURCE_DIR}/seeding/triplet_finding.cu
  ${TRACCC_CUDA_ALGORITHMS_SOURCE_DIR}/seeding/weight_updating.cu
  ${TRACCC_CUDA_ALGORITHMS_SOURCE_DIR}/seeding/seed_selecting.cu)

target_include_directories(traccc_cuda_algorithms
  PUBLIC
  $<BUILD_INTERFACE:${TRACCC_CUDA_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  )

target_compile_options(traccc_cuda_algorithms
  PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
  -arch=sm_52 
  -gencode=arch=compute_52,code=sm_52 
  -gencode=arch=compute_60,code=sm_60  
  -gencode=arch=compute_61,code=sm_61  
  -gencode=arch=compute_70,code=sm_70  
  -gencode=arch=compute_75,code=sm_75 
  -gencode=arch=compute_80,code=sm_80 
  -gencode=arch=compute_80,code=compute_80 
  --expt-relaxed-constexpr
  -fmad=false>)

target_link_libraries(traccc_cuda_algorithms PUBLIC traccc::core vecmem::core vecmem::cuda)

add_library(traccc::cuda::algorithms ALIAS traccc_cuda_algorithms)
