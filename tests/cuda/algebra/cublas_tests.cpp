/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <gtest/gtest.h>

// vecmem
#include <vecmem/memory/cuda/managed_memory_resource.hpp>
#include "vecmem/memory/cuda/device_memory_resource.hpp"
#include "vecmem/memory/cuda/host_memory_resource.hpp"
#include <vecmem/memory/host_memory_resource.hpp>
#include <vecmem/containers/vector.hpp>
#include "vecmem/utils/cuda/copy.hpp"

// cuda
#include <cublas_v2.h>
#include <cuda_runtime.h>

// acts
#include <Acts/Definitions/TrackParametrization.hpp>

// This defines the local frame test suite
TEST(algebra, cublas_tests) {    

    // declare cublas objects
    cublasHandle_t m_handle;
    cublasStatus_t m_status;

    // create cublas handler
    m_status = cublasCreate(&m_handle);

    // alpha beta for cublas option
    const double alpha = 1;
    const double beta = 0;
    
    // generate test matrices    
    Acts::BoundSymMatrix A = Acts::BoundSymMatrix::Random();
    Acts::BoundSymMatrix B = Acts::BoundSymMatrix::Random();
    Acts::BoundSymMatrix C = A*B;
    
    std::cout << "C truth matrix" << std::endl;
    std::cout << C << std::endl;
    
    /*-----------------------------------------
      Test case 1) A:device B:device C:device
      -----------------------------------------*/

    // memory copy helper
    vecmem::cuda::copy m_copy;
    
    // The host/device memory resources
    vecmem::cuda::device_memory_resource dev_mr;
    vecmem::cuda::host_memory_resource host_mr;

    vecmem::vector<Acts::BoundSymMatrix> A_host(1,&host_mr);
    vecmem::vector<Acts::BoundSymMatrix> B_host(1,&host_mr);
    vecmem::vector<Acts::BoundSymMatrix> C_host(1,&host_mr);

    A_host[0] = A;
    B_host[0] = B;

    // copy matrix from host to device
    auto A_dev = m_copy.to ( vecmem::get_data( A_host ), dev_mr, vecmem::copy::type::host_to_device );
    auto B_dev = m_copy.to ( vecmem::get_data( B_host ), dev_mr, vecmem::copy::type::host_to_device );
    auto C_dev = m_copy.to ( vecmem::get_data( C_host ), dev_mr, vecmem::copy::type::host_to_device );

    // Do the matrix multiplication
    m_status = cublasDgemm(m_handle,
			   CUBLAS_OP_N, CUBLAS_OP_N,
			   A.rows(), B.cols(), B.rows(),
			   &alpha,
			   &A_dev.ptr()[0](0), A.rows(),
			   &B_dev.ptr()[0](0), B.rows(),
			   &beta, 
			   &C_dev.ptr()[0](0), C.rows()
			   );

    // retrieve the result to the host
    m_copy( C_dev, C_host, vecmem::copy::type::device_to_host );

    std::cout << std::endl;
    std::cout << "Test case 1) A:device B:device C:device" << std::endl;
    std::cout << C_host[0] << std::endl;              

    for( std::size_t i = 0; i < C.rows()*C.cols(); ++i ) {
	EXPECT_TRUE( abs(C_host[0](i) - C(i)) < 1e-8 );
    }
    
    /*--------------------------------------------
      Test case 2) A:unified B:unified C:unified
      --------------------------------------------*/
    
    // The managed memory resource
    vecmem::cuda::managed_memory_resource mng_mr;

    // matrix
    vecmem::vector<Acts::BoundSymMatrix> A_mng(1,&mng_mr);
    vecmem::vector<Acts::BoundSymMatrix> B_mng(1,&mng_mr);
    vecmem::vector<Acts::BoundSymMatrix> C_mng(1,&mng_mr);

    A_mng[0] = A;
    B_mng[0] = B;
    C_mng[0] = Acts::BoundSymMatrix::Zero();
    
    m_status = cublasDgemm(m_handle,
			   CUBLAS_OP_N, CUBLAS_OP_N,
			   A.rows(), B.cols(), B.rows(),
			   &alpha,
			   &A_mng[0](0), A.rows(),
			   &B_mng[0](0), B.rows(),
			   &beta, 
			   &C_mng[0](0), C.rows()
			   );

    std::cout << std::endl;
    std::cout << "Test case 2) A:unified B:unified C:unified (Does NOT work correctly)" << std::endl;    
    std::cout << C_mng[0] << std::endl;       
    

    /*--------------------------------------------
      Test case 3) A:unified B:unified C:device
      --------------------------------------------*/

    C_host[0] = Acts::BoundSymMatrix::Zero();
    C_dev = m_copy.to ( vecmem::get_data( C_host ), dev_mr, vecmem::copy::type::host_to_device );
    
    m_status = cublasDgemm(m_handle,
			   CUBLAS_OP_N, CUBLAS_OP_N,
			   A.rows(), B.cols(), B.rows(),
			   &alpha,
			   &A_mng[0](0), A.rows(),
			   &B_mng[0](0), B.rows(),
			   &beta, 
			   &C_dev.ptr()[0](0), C.rows()
			   );

    m_copy( C_dev, C_host, vecmem::copy::type::device_to_host );    
    
    std::cout << std::endl;
    std::cout << "Test case 3) A:unified B:unified C:device" << std::endl;    
    std::cout << C_host[0] << std::endl;       

    for( std::size_t i = 0; i < C.rows()*C.cols(); ++i ) {
	EXPECT_TRUE( abs(C_host[0](i) -C(i))<1e-8 );
    }

    /*--------------------------------------------
      Test case 4) A:device B:device C:unified
      --------------------------------------------*/

    C_mng[0] = Acts::BoundSymMatrix::Zero();
    
    m_status = cublasDgemm(m_handle,
			   CUBLAS_OP_N, CUBLAS_OP_N,
			   A.rows(), B.cols(), B.rows(),
			   &alpha,
			   &A_dev.ptr()[0](0), A.rows(),
			   &B_dev.ptr()[0](0), B.rows(),
			   &beta, 
			   &C_mng[0](0), C.rows()
			   );

    std::cout << std::endl;
    std::cout << "Test case 4) A:device B:device C:unified (Does NOT work correctly)" << std::endl;    
    std::cout << C_mng[0] << std::endl;       
        
    // destroy cublas handler
    m_status = cublasDestroy(m_handle);
}

// Google Test can be run manually from the main() function
// or, it can be linked to the gtest_main library for an already
// set-up main() function primed to accept Google Test test cases.
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
