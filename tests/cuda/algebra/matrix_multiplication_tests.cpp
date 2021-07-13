/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
// This defines the local frame test suite

#include <gtest/gtest.h>

// vecmem
#include <vecmem/memory/cuda/managed_memory_resource.hpp>
#include "vecmem/memory/cuda/device_memory_resource.hpp"
#include "vecmem/memory/cuda/host_memory_resource.hpp"
#include <vecmem/memory/host_memory_resource.hpp>
#include <vecmem/containers/vector.hpp>
#include "vecmem/utils/cuda/copy.hpp"

// acts
#include <Acts/Definitions/TrackParametrization.hpp>

// std
#include <chrono>

#include "matmul_kernel.cuh"

TEST(algebra, matrix_multiplcation_tests) {

    // batch_size (number of matrices)
    const int batch_size = 10000;

    // matrix type
    using matrix_t = Acts::FreeSymMatrix; // 8x8 matrix
    //using matrix_t = Acts::BoundSymMatrix; // 6x6 matrix
    
    // for timing benchmark
    float cpu_managed(0);
    float gpu_managed(0);

    float cpu_host(0);
    float gpu_device(0); 
    
    /*---------------------------------------
      Part 1: Managed memory benchmark
      ---------------------------------------*/
        
    // The host/device memory resources
    vecmem::cuda::managed_memory_resource mng_mr;

    // generate test matrices        
    vecmem::vector< matrix_t > A(batch_size, &mng_mr);
    vecmem::vector< matrix_t > B(batch_size, &mng_mr);

    // Fill the matrices
    for (size_t i_b = 0; i_b < batch_size; i_b++){
	A[i_b] = matrix_t::Random();
	B[i_b] = matrix_t::Random();
    }
    
    /*-------------------------------------------
      1.A simple cpu eigen matrix multiplication
      -------------------------------------------*/

    vecmem::vector< matrix_t > C_cpu(batch_size, &mng_mr);

    /*time*/ auto start_cpu_managed = std::chrono::system_clock::now();
    
    for (size_t i_b = 0; i_b < batch_size; i_b++){
	C_cpu[i_b] = A[i_b]*B[i_b];
    }

    /*time*/ auto end_cpu_managed = std::chrono::system_clock::now();
    /*time*/ std::chrono::duration<double> time_cpu_managed = end_cpu_managed - start_cpu_managed;
    /*time*/ cpu_managed += time_cpu_managed.count();

    /*---------------------------------------------
      1.B simple cuda kernel matrix multiplication 
      ---------------------------------------------*/

    vecmem::vector< matrix_t > C_gpu(batch_size, &mng_mr);
    
    matmul(batch_size,
	   vecmem::get_data( A ),
	   vecmem::get_data( B ),
	   vecmem::get_data( C_gpu ),
	   gpu_managed);
    
    // Compare the cpu and gpu result
    for (size_t i_b = 0; i_b < batch_size; i_b++){	
	for(size_t i_m = 0; i_m < C_cpu[0].rows()*C_cpu[0].cols(); ++i_m ) {	    	    
	    EXPECT_TRUE( abs(C_cpu[i_b](i_m) - C_gpu[i_b](i_m)) < 1e-8 );
	}
    }

    /*---------------------------------------
      Part 2: Host/Device memory benchmark
      ---------------------------------------*/

    // The host/device memory resources
    vecmem::cuda::host_memory_resource host_mr;	
    vecmem::cuda::device_memory_resource dev_mr;
    
    // generate test matrices        
    vecmem::vector< matrix_t > A_host(batch_size, &host_mr);
    vecmem::vector< matrix_t > B_host(batch_size, &host_mr);
    vecmem::vector< matrix_t > C_host(batch_size, &host_mr);

    // Copy the matrix
    for (size_t i_b = 0; i_b < batch_size; i_b++){
	A_host[i_b] = A[i_b];
	B_host[i_b] = B[i_b];	
    }

    /*time*/ auto start_cpu_host = std::chrono::system_clock::now();
    
    for (size_t i_b = 0; i_b < batch_size; i_b++){
	C_cpu[i_b] = A_host[i_b]*B_host[i_b];
    }

    /*time*/ auto end_cpu_host = std::chrono::system_clock::now();
    /*time*/ std::chrono::duration<double> time_cpu_host = end_cpu_host - start_cpu_host;
    /*time*/ cpu_host += time_cpu_host.count();
    
    // memory copy helper
    vecmem::cuda::copy m_copy;
    
    // Transfer data from host to device
    auto A_dev = m_copy.to ( vecmem::get_data( A_host ), dev_mr, vecmem::copy::type::host_to_device);
    auto B_dev = m_copy.to ( vecmem::get_data( B_host ), dev_mr, vecmem::copy::type::host_to_device);
    auto C_dev = m_copy.to ( vecmem::get_data( C_host ), dev_mr, vecmem::copy::type::host_to_device);

    matmul(batch_size,
	   A_dev, B_dev, C_dev,
	   gpu_device);

    // retrieve the result to the host
    for (size_t i_b = 0; i_b < batch_size; i_b++){	   
	m_copy( C_dev, C_gpu, vecmem::copy::type::device_to_host );
    }
    
    // Compare the cpu and gpu result
    for (size_t i_b = 0; i_b < batch_size; i_b++){	
	for(size_t i_m = 0; i_m < C_cpu[0].rows()*C_cpu[0].cols(); ++i_m ) {	    	    
	    EXPECT_TRUE( abs(C_cpu[i_b](i_m) - C_gpu[i_b](i_m)) < 1e-8 );
	}
    }

    
    std::cout << "==> Elpased time ... " << std::endl;
    std::cout << "cpu managed time: " << cpu_managed << std::endl;
    std::cout << "gpu managed time: " << gpu_managed << std::endl;
    std::cout << "mat-mul speedup: " << cpu_managed/gpu_managed << std::endl;
    std::cout << std::endl;
    
    std::cout << "cpu host time: " << cpu_host << std::endl;
    std::cout << "gpu device time: " << gpu_device << std::endl;
    std::cout << "mat-mul speedup: " << cpu_host/gpu_device << std::endl;    
}

// Google Test can be run manually from the main() function
// or, it can be linked to the gtest_main library for an already
// set-up main() function primed to accept Google Test test cases.
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}

