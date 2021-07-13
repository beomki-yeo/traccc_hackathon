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
#include "vecmem/utils/cuda/copy.hpp"

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
    template< int n_rows, int n_cols >
    struct internal_matrix{
	// constructor for internal matrix
	internal_matrix():
	    n_size(n_rows*n_cols),
	    mat_host(vecmem::vector<scalar_t>(n_rows*n_cols*batch_size, 0, &host_mr)),
	    mat_dev(vecmem::data::vector_buffer<scalar_t>(n_rows*n_cols*batch_size, dev_mr))
	{

	    //mat_dev = m_copy.to ( vecmem::get_data( mat_host ), dev_mr, vecmem::copy::type::host_to_device);
	    
	    for (unsigned int i_b=0; i_b<batch_size; i_b++){
		bptr[i_b] = mat_dev.ptr() + (i_b*n_rows*n_cols);
	    }	    
	}

	scalar_t** get_bptr(){
	    return bptr;
	}

	void dev2host(){
	    m_copy( mat_dev, mat_host, vecmem::copy::type::device_to_host );
	}
       	
	// memory copy helper
	vecmem::cuda::copy m_copy;
	
	// The host/device memory resources
	vecmem::cuda::device_memory_resource dev_mr;
	vecmem::cuda::host_memory_resource host_mr;
	
	scalar_t* bptr[batch_size];
	vecmem::vector<scalar_t> mat_host;
	vecmem::data::vector_buffer<scalar_t> mat_dev;
	
	int n_size;
    };
    */
    
    // kalman update
    void update(scalar_t** meas_array,
		scalar_t** proj_array,
		scalar_t** pred_vector_array,
		scalar_t** pred_cov_array) {
	
	scalar_t alpha, beta;	
	alpha = 1;
	beta = 0;
	
	/*
	m_status = cublasGgemmBatched(m_handle,
				      CUBLAS_OP_N, CUBLAS_OP_N,
				      meas_dim, params_dim, params_dim,
				      &alpha,
				      proj_array, meas_dim,
				      pred_cov_array, params_dim,
				      &beta,
				      Cptr, meas_dim,
				      batch_size
				      );		
	*/

	// cuda error check
	CUDA_ERROR_CHECK(cudaGetLastError());
	CUDA_ERROR_CHECK(cudaDeviceSynchronize());
	
    }

private:
    // The host/device memory resources
    vecmem::cuda::device_memory_resource dev_mr;
    vecmem::cuda::host_memory_resource host_mr;
    
    // internal matrices
    //internal_matrix<2,6> HC; // H(2x6) * C(6x6)
    //internal_matrix<2,2> C2; // H(2x6) * C(6x6) * H^T(6x2) + R(2x2)
    //internal_matrix<2,2> C2inv; // C2^inv    
    //internal_matrix<6,2> HC2inv; // H^T(6x2) * C2inv(2x2)
    //internal_matrix<6,2> K; // C(6x6) * H^T(6x2) * C2inv(2x2)
    //internal_matrix<6,1> gain; // K(6x2) * residual (2x1)
    //internal_matrix<6,6> I_KH; // I(6x6) - K(6x2)*H(2x6)
    
    // cublas objects
    cublasHandle_t m_handle;
    cublasStatus_t m_status;    
};

}  // namespace cuda
}  // namespace traccc
