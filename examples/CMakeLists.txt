add_subdirectory(cpu)
if (TRACCC_BUILD_cuda)
  add_subdirectory(cuda)
endif()
