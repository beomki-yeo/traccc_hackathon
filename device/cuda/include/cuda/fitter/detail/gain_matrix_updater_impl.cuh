/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/fitter/detail/cublas_wrapper.hpp>
#include <cuda/utils/definitions.hpp>

namespace traccc {
namespace cuda {

template < typename scalar_t,
	   int meas_dim,
	   int params_dim,
	   int batch_size >    
class gain_matrix_updater_impl{
public:
    
    // constructor: create the cublas status and initialize matrix   
    gain_matrix_updater_impl(){
	m_status = cublasCreate(&m_handle);
    }
    
    // destructor: destruct the cublas status
    ~gain_matrix_updater_impl(){
	m_status = cublasDestroy(m_handle);
    }

    // matrices for cublas calculation
    /*
    template< int n_row, int n_col >
    struct internal_matrix{
	// constructor for internal matrix
	internal_matrix(){
	    n_size = n_row*n_col;
	    CUDA_ERROR_CHECK (cudaMallocManaged((void**)&mat, n_size*sizeof(scalar_t)) );
	    for (int i=0; i<20; i++){
		mat[i] = 1.;
	    }
	    
	    //set_zero();
	}
	
	// destructor for internal matrix	
	~internal_matrix(){
	    CUDA_ERROR_CHECK( cudaFree(mat) );
	}
	
	// make all elements zero
	
	void set_zero(){
	    scalar_t a = 1.;
	    cudaMemset(mat, a, n_size*sizeof(scalar_t));
	}
	
	scalar_t* mat;
	int n_size;
    };
    */
    
    // matrices for cublas calculation
    template< int n_row, int n_col >
    struct internal_matrix{
	// constructor for internal matrix
	internal_matrix(){
	    n_size = n_row*n_col;
	    for (int i_b=0; i_b<batch_size; i_b++){
		cudaMallocManaged(&mat[i_b], n_size*sizeof(scalar_t));
	    }
	}
	
	// destructor for internal matrix	
	~internal_matrix(){
	    for (int i_b=0; i_b<batch_size; i_b++){
		cudaFree(mat[i_b]);
	    }
	}
	
	// make all elements zero
	void set_zero(){
	    //cudaMemset(&mat, 0, batch_size*n_size*sizeof(scalar_t));
	}
	
	scalar_t* mat[batch_size];
	int n_size;
    };
    
    // kalman update
    void update(const scalar_t** meas_array,
		const scalar_t** proj_array,
		scalar_t** proj2_array,	
		const scalar_t** pred_vector_array,
		const scalar_t** pred_cov_array) {
	
	scalar_t alpha, beta;	
	alpha = 1;
	beta = 0;
	
	m_status = cublasDgemm(m_handle,
			       CUBLAS_OP_N, CUBLAS_OP_N,
			       meas_dim, params_dim, params_dim,
			       &alpha,
			       proj_array[0], meas_dim,
			       pred_cov_array[0], params_dim,
			       &beta, 
			       proj2_array[0], meas_dim
			       );
	       	
	/*
	m_status = cublasGgemmBatched(m_handle,
				      CUBLAS_OP_N, CUBLAS_OP_N,
				      meas_dim, params_dim, params_dim,
				      &alpha,
				      proj_array, meas_dim,
				      pred_cov_array, params_dim,
				      &beta, 
				      proj2_array, meas_dim,
				      1
				      );		
	*/
	for (int i=0; i<12; i++){
	    std::cout << *(proj_array[0]+i) << std::endl;
	}

	for (int i=0; i<36; i++){
	    std::cout << *(pred_cov_array[0]+i) << std::endl;
	}
	
	
	for (int i=0; i<12; i++){
	    std::cout << *(proj2_array[0]+i) << std::endl;
	}
	
	// cuda error check
	CUDA_ERROR_CHECK(cudaGetLastError());
	CUDA_ERROR_CHECK(cudaDeviceSynchronize());
	
    }

private:
    
    // internal matrices
    internal_matrix<2,6> HC; // H(2x6) * C(6x6)
    //internal_matrix<2,2> C2; // H(2x6) * C(6x6) * H^T(6x2) + R(2x2)
    //internal_matrix<2,2> C2inv; // C2^inv    
    //internal_matrix<6,2> HC2inv; // H^T(6x2) * C2inv(2x2)
    //internal_matrix<6,2> K; // C(6x6) * H^T(6x2) * C2inv(2x2)
    //internal_matrix<6,1> gain; // K(6x2) * residual (2x1)
    //internal_matrix<6,6> I_KH; // I(6x6) - K(6x2)*H(2x6)
    
    // cublas objects
    cublasHandle_t m_handle;
    cublasStatus_t m_status;
    //cublasOperation_t m_op_normal = CUBLAS_OP_N;
    //cublasOperation_t m_op_transpose = CUBLAS_OP_T;
    
};

}  // namespace cuda
}  // namespace traccc
