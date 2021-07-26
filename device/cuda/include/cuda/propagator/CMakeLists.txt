add_library(traccc_cuda_propagator SHARED
  ${TRACCC_CUDA_PROPAGATOR_SOURCE_DIR}/eigen_stepper.cu
  ${TRACCC_CUDA_PROPAGATOR_SOURCE_DIR}/direct_navigator.cu
  ${TRACCC_CUDA_PROPAGATOR_SOURCE_DIR}/propagator.cu)

target_include_directories(traccc_cuda_propagator
  PUBLIC
  $<BUILD_INTERFACE:${TRACCC_CUDA_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  )

target_compile_options(traccc_cuda_propagator
  PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
  --expt-relaxed-constexpr
  #--fmad=false
  >)

target_link_libraries(traccc_cuda_propagator PUBLIC traccc::core CUDA::cudart CUDA::cublas)

add_library(traccc::cuda::propagator ALIAS traccc_cuda_propagator)
