add_executable (seq_example_cuda seq_example_cuda.cpp)
add_executable (seeding_example_cuda seeding_example_cuda.cpp)

target_link_libraries (seq_example_cuda LINK_PUBLIC
  traccc::io
  traccc::core
  traccc::cuda
  traccc::algorithms)

target_link_libraries (seeding_example_cuda LINK_PUBLIC
  traccc::io
  traccc::core
  traccc::cuda
  traccc::algorithms)

