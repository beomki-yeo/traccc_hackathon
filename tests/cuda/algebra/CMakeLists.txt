add_traccc_test(cublas_tests cublas_tests.cpp traccc::core traccc::cuda)
add_traccc_test(batched_cublas_matmul_tests batched_cublas_matmul_tests.cpp traccc::core traccc::cuda)
add_traccc_test(matmul_tests
  "matmul_tests.cpp;matmul_kernel.cuh;matmul_kernel.cu"
  "traccc::core;traccc::cuda")
