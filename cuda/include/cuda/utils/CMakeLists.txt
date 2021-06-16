add_library(traccc_cuda_utils SHARED
    ${TRACCC_CUDA_UTILS_SOURCE_DIR}/warming_up.cu)

target_include_directories(traccc_cuda_utils
  PUBLIC
  $<BUILD_INTERFACE:${TRACCC_CUDA_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  )

target_compile_options(traccc_cuda_utils
  PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:>)

target_link_libraries(traccc_cuda_utils PUBLIC traccc::core vecmem::core vecmem::cuda)


add_library(traccc::cuda::utils ALIAS traccc_cuda_utils)
