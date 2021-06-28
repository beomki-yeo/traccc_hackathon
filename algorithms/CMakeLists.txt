add_library(traccc_algorithms INTERFACE)

target_include_directories(traccc_algorithms
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_include_directories(traccc_algorithms INTERFACE "$ENV{ACTS_HOME}/Examples/Framework/include")
target_include_directories(traccc_algorithms INTERFACE "$ENV{ACTS_HOME}/Examples/Detectors/MagneticField/include")

target_link_libraries(traccc_algorithms
  INTERFACE vecmem::core ActsCore)

install(TARGETS traccc_algorithms
        PUBLIC_HEADER
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/algorithms)

install(
  DIRECTORY include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

add_library(traccc::algorithms ALIAS traccc_algorithms)

