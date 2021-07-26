/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace traccc {
namespace cuda {

// batched matrix multiplication for float
static inline cublasStatus_t cublasGgemmBatched(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const float *alpha, float **A, int lda, float **B,
    int ldb, const float *beta, float **C, int ldc, int batch_size) {
    return cublasSgemmBatched(handle, transa, transb, m, n, k, alpha, A, lda, B,
                              ldb, beta, C, ldc, batch_size);
}

// batched matrix multiplication for double
static inline cublasStatus_t cublasGgemmBatched(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const double *alpha, double **A, int lda, double **B,
    int ldb, const double *beta, double **C, int ldc, int batch_size) {
    return cublasDgemmBatched(handle, transa, transb, m, n, k, alpha, A, lda, B,
                              ldb, beta, C, ldc, batch_size);
}

}  // namespace cuda
}  // namespace traccc
