cmake_minimum_required(VERSION 3.12)
project(kokkos-resilience VERSION 0.1.0)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")

add_library(resilience)
add_library(Kokkos::resilience ALIAS resilience)

include_directories(${kokkos-resilience_SOURCE_DIR})
include_directories(${kokkos-resilience_BINARY_DIR})

    # Require C++14
if (CMAKE_CXX_COMPILER MATCHES ".*nvcc_wrapper" )
    message("nvcc wrapper requires that CXX standard by set in kokkos build")
else()
   target_compile_features(resilience PUBLIC cxx_std_14)
endif()

target_include_directories(resilience PUBLIC
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/resilience>
                           $<INSTALL_INTERFACE:include>
                           )

# TODO: change this when Kokkos cmake gets updated
find_package(Kokkos REQUIRED NO_CMAKE_PACKAGE_REGISTRY)
##target_include_directories(kokkos INTERFACE ${Kokkos_INCLUDE_DIRS})
##message("Kokkos include dirs: ${Kokkos_INCLUDE_DIRS}")

##target_link_libraries(resilience PUBLIC kokkos)
TARGET_LINK_KOKKOS(resilience PUBLIC)

option( USE_VELOC "use VeloC backend for automatic checkpointing" ON )


# VeloC backend
if (USE_VELOC)
   find_package(VeloC REQUIRED)
   target_link_libraries(resilience PRIVATE veloc::veloc)
endif()


# Library options
option(KR_ENABLE_TRACING "Enable tracing of resilience functions" OFF)
if (KR_ENABLE_TRACING)
  target_compile_definitions(resilience PUBLIC KR_ENABLE_TRACING)
endif()

option( USE_HDF5 "add HDF5 support" OFF )
option( USE_HDF5_PARALLEL "use parallel version of HDF5" OFF )

if (USE_HDF5_PARALLEL)
  set(USE_HDF5 ON)
  set(KOKKOS_ENABLE_HDF5_PARALLEL ON)
endif()

if (USE_HDF5)
   find_package(HDF5 REQUIRED)
   target_link_libraries(resilience PRIVATE HDF5::HDF5)
   set(KOKKOS_ENABLE_HDF5 ON)
endif()

# MPI requirement
if (USE_VELOC OR USE_HDF5_PARALLEL)
   find_package(MPI REQUIRED)
   target_link_libraries(resilience PRIVATE MPI::MPI_CXX)
   set(KOKKOS_ENABLE_VELOC ON)
endif()

if (USE_VELOC)
   link_directories(${KOKKOS_LIBRARY_DIRS} ${VeloC_LINK_DIRECTORIES})
else()
   link_directories(${KOKKOS_LIBRARY_DIRS})
endif()


find_package(Boost REQUIRED)

target_link_libraries(resilience PUBLIC Boost::boost)

add_subdirectory(src)

# Export targets for in-tree linking
export(TARGETS resilience
       NAMESPACE Kokkos::
       FILE resilienceTargets.cmake
       )

if (Kokkos_ENABLE_Cuda)
   set(KOKKOS_ENABLE_RES_CUDA ON)
else()
   set(KOKKOS_ENABLE_RES_CUDA OFF)
endif()

# Set up find_package config and version file
# Allow same major version compatibility
# Possibly in the future find_package(Kokkos REQUIRED COMPONENTS reslience)
include(CMakePackageConfigHelpers)
configure_package_config_file(cmake/resilienceConfig.cmake.in
                              ${CMAKE_CURRENT_BINARY_DIR}/resilienceConfig.cmake
                              INSTALL_DESTINATION ${CMAKE_INSTALL_PREFIX}/cmake)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/resilienceConfigVersion.cmake
                                 COMPATIBILITY SameMajorVersion
                                 )

# Set install rules
install(TARGETS resilience EXPORT resilienceTargets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include
        )

install(EXPORT resilienceTargets
        FILE resilienceTargets.cmake
        NAMESPACE Kokkos::
        DESTINATION cmake
        )


install(FILES ${CMAKE_CURRENT_BINARY_DIR}/resilienceConfig.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/resilienceConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_PREFIX}/cmake )


if (USE_VELOC)
   install(FILES
        ${CMAKE_MODULE_PATH}/FindVeloC.cmake
        DESTINATION ${CMAKE_INSTALL_PREFIX}/cmake/Modules )

   # Need to install/export VeloC find module for downstream dependencies
   configure_file(cmake/Modules/FindVeloC.cmake Modules/FindVeloC.cmake COPYONLY)

endif()

if (USE_HDF5)
   install(FILES
        ${CMAKE_MODULE_PATH}/FindHDF5.cmake
        DESTINATION ${CMAKE_INSTALL_PREFIX}/cmake/Modules )

endif()

CONFIGURE_FILE(cmake/KokkosResilience_config.h.in KokkosResilience_config.h @ONLY)
INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/KokkosResilience_config.h DESTINATION ${CMAKE_INSTALL_PREFIX}/include)

# Add subdirectories for examples and tests if they are enabled
option(KR_ENABLE_TESTS "Enable tests in the build" ON)
option(KR_ENABLE_EXAMPLES "Enable examples in the build" ON)

add_subdirectory(tpl)

if (KR_ENABLE_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if (KR_ENABLE_EXAMPLES)
  add_subdirectory(examples)
endif()