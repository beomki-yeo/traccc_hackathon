add_traccc_test(gain_maxtrix_updater gain_matrix_updater_tests.cpp traccc::core traccc::cuda)
add_traccc_test(cublas_tests cublas_tests.cpp traccc::core traccc::cuda)
add_traccc_test(batched_cublas_tests batched_cublas_tests.cpp traccc::core traccc::cuda)
add_traccc_test(matrix_multiplication_tests
  "matrix_multiplication_tests.cpp;matmul_kernel.cuh;matmul_kernel.cu"
  "traccc::core;traccc::cuda")
