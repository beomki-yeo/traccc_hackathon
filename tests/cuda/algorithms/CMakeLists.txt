add_traccc_test(gain_maxtrix gain_matrix_tests.cpp traccc::core traccc::cuda)
add_traccc_test(covariance_engine covariance_engine_tests.cpp traccc::core traccc::cuda)
add_traccc_test(propagator propagator_tests.cpp traccc::core traccc::cuda traccc::io)
add_traccc_test(stepper stepper_tests.cpp traccc::core traccc::cuda traccc::io)
#add_traccc_test(rk4 rk4_tests.cpp traccc::core traccc::cuda traccc::io)
#add_traccc_test(covariance_transport covariance_transport_tests.cpp traccc::core traccc::cuda traccc::io)
#add_traccc_test(direct_navigator direct_navigator_tests.cpp traccc::core traccc::cuda traccc::io)
