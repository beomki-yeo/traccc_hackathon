add_library(traccc_cuda_fitter INTERFACE)

target_include_directories(traccc_cuda_fitter
  INTERFACE
  $<BUILD_INTERFACE:${TRACCC_CUDA_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  )

target_compile_options(traccc_cuda_fitter
  INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:  
  --fmad=false>)

target_link_libraries(traccc_cuda_fitter INTERFACE traccc::core CUDA::cudart CUDA::cublas)

add_library(traccc::cuda::fitter ALIAS traccc_cuda_fitter)
