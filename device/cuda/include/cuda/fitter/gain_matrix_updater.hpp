/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

/* Includes, cuda */
#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace traccc {
namespace cuda {

class gain_matrix_updater{

    gain_matrix_updater(int dim, int batch_size){
	m_status = cublasCreate(&m_handle);

    }
    ~gain_matrix_updater(){
	m_status = cublasDestroy(m_handle);
    }
    
    void update(double** meas_array,
		double** proj_array,
		double** pred_par_array,
		double** pred_cov_array) const {
	
	m_status = cublasDgemmBatched(m_handle,
				      CUBLAS_OP_N, CUBLAS_OP_N,
				      m_dim, m_dim, m_dim, // matrix dimension
				      1, // alpha
				      proj_array, m_dim,
				      proj_array, m_dim,
				      1, // beta
				      proj_array, m_dim,
				      m_batch_size // batch_size
				      );
	
    }

private:
    // matrix dimension    
    int m_dim; 
    // batch size    
    int m_batch_size; 
    // cublas objects
    cublasHandle_t m_handle;
    cublasStatus_t m_status;    
};


}  // namespace cuda
}  // namespace traccc

