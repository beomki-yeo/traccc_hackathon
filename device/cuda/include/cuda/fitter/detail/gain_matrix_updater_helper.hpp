/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda/fitter/detail/cublas_wrapper.hpp>

namespace traccc {
namespace cuda {

template < typename scalar_t, int dim, int batch_size >    
class gain_matrix_updater_helper{

    // constructor: create the cublas status
    gain_matrix_updater_helper(){
	m_status = cublasCreate(&m_handle);
    }

    // destructor: destruct the cublas status
    ~gain_matrix_updater_helper(){
	m_status = cublasDestroy(m_handle);
    }

    // matrices for cublas calculation
    struct internal_matrix{

	// constructor for internal matrix
	internal_matrix(){
	    for (int i_b=0; i_b<batch_size; i_b++){
		cudaMallocManaged(&mat[i_b], dim*dim*sizeof(scalar_t));
	    }
	    set_zero(mat);
	}

	// destructor for internal matrix	
	~internal_matrix(){
	    for (int i_b=0; i_b<batch_size; i_b++){
		cudaFree(mat[i_b]);
	    }	    
	}

	// make all elements zero
	void set_zero(){
	    for (int i_b=0; i_b<batch_size; i_b++){
		cudaMemset(mat[i_b], 0, dim*dim*sizeof(int));
	    }	   
	}
	
	float* mat[batch_size];
    };

    // kalman update
    void update(scalar_t** meas_array,
		scalar_t** proj_array,
		scalar_t** pred_par_array,
		scalar_t** pred_cov_array) {	
	gemm_batched(m_op_normal,
		     m_op_transpose,
		     pred_cov_array,
		     proj_array,
		     M1);
    }


    // batched matrix multiplication     
    void gemm_batched(cublasOperation_t opA,
		      cublasOperation_t opB,
		      scalar_t** A,
		      scalar_t** B,
		      scalar_t** C){
	const scalar_t alpha = 1;
	const scalar_t beta = 1;	
	
	m_status = cublasGgemmBatched(m_handle,
				      opA, opB,
				      dim, dim, dim, // matrix dimension
				      &alpha, // alpha
				      A, dim,
				      B, dim,
				      &beta, // beta
				      C, dim,
				      batch_size // batch_size
				      );	
    }

    // batched matrix inversion
    void inv_batched(scalar_t** A){

    }
        
private:
    // internal matrices
    internal_matrix M1;
    internal_matrix M2;
    internal_matrix M3;
    internal_matrix M4;        
    
    // cublas objects
    cublasHandle_t m_handle;
    cublasStatus_t m_status;
    cublasOperation_t m_op_normal = CUBLAS_OP_N;
    cublasOperation_t m_op_transpose = CUBLAS_OP_T;
    
};

}  // namespace cuda
}  // namespace traccc
