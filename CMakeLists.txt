cmake_minimum_required(VERSION 3.17)
project(kokkos-resilience VERSION 0.1.0)

INCLUDE(GNUInstallDirs)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")

add_library(resilience)
add_library(Kokkos::resilience ALIAS resilience)


option(KR_ALL_WARNINGS "Enable all warnings" ON)
option(KR_WARNINGS_AS_ERRORS "Enable warnings as errors" ON)

if (KR_ALL_WARNINGS)
  if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(resilience PRIVATE "-Wall")
  endif()
endif()

if (KR_WARNINGS_AS_ERRORS)
  if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(resilience PRIVATE "-Werror")
  endif()
endif()

target_include_directories(resilience PUBLIC
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/src>
                           $<INSTALL_INTERFACE:include>
                           )

find_package(Kokkos 4.0 REQUIRED)

set_property(TARGET resilience PROPERTY CXX_STANDARD ${Kokkos_CXX_STANDARD})

target_link_libraries(resilience PUBLIC Kokkos::kokkos)

option(KR_ENABLE_VELOC "use VeloC backend for automatic checkpointing" ON)
option(KR_ENABLE_STDFILE "use StdFile backend for automatic checkpointing" ON)

include(CMakeDependentOption)

cmake_dependent_option(KR_VELOC_BAREBONE "Use the barebone branch of VeloC" OFF "KR_ENABLE_VELOC" OFF)

# VeloC backend
if (KR_ENABLE_VELOC)
  find_package(veloc REQUIRED)
  target_link_libraries(resilience PUBLIC veloc::client)
  target_compile_definitions(resilience PUBLIC KR_ENABLE_VELOC)
  set(KR_ENABLE_MPI_BACKENDS ON)

  if (KR_VELOC_BAREBONE)
    target_compile_definitions(resilience PRIVATE KR_VELOC_BAREBONE)
  endif()
endif()

# StdFile backend
if (KR_ENABLE_STDFILE)
  target_compile_definitions(resilience PUBLIC KR_ENABLE_STDFILE)
endif()

if (KR_ENABLE_MPI_BACKENDS)
  target_compile_definitions(resilience PUBLIC KR_ENABLE_MPI_BACKENDS)
endif()

# Library options
option(KR_ENABLE_TRACING "Enable tracing of resilience functions" OFF)
if (KR_ENABLE_TRACING)
  target_compile_definitions(resilience PUBLIC KR_ENABLE_TRACING)
endif()

option( KR_ENABLE_STDIO "use stdio for manual checkpoint" OFF )
option( KR_ENABLE_HDF5 "add HDF5 support" OFF )
option( KR_ENABLE_HDF5_PARALLEL "use parallel version of HDF5" OFF )

if (KR_ENABLE_STDIO)
  set(KR_ENABLE_MANUAL_CHECKPOINT ON)
endif()

if (KR_ENABLE_HDF5_PARALLEL)
  set(KR_ENABLE_HDF5 ON)
  target_compile_definitions(resilience PUBLIC KR_ENABLE_HDF5_PARALLEL)
  set(KR_ENABLE_MANUAL_CHECKPOINT ON)
endif()

if (KR_ENABLE_HDF5)
   find_package(HDF5 REQUIRED)
   target_link_libraries(resilience PUBLIC HDF5::HDF5)
   target_compile_definitions(resilience PUBLIC KR_ENABLE_HDF5)
   set(KR_ENABLE_MANUAL_CHECKPOINT ON)
endif()

if (KR_ENABLE_MANUAL_CHECKPOINT)
  target_compile_definitions(resilience PUBLIC KR_ENABLE_MANUAL_CHECKPOINT)
endif()

# MPI requirement
if (KR_ENABLE_VELOC OR KR_ENABLE_HDF5_PARALLEL)
   find_package(MPI REQUIRED)
   target_link_libraries(resilience PRIVATE MPI::MPI_CXX)
endif()

# Devices
list(FIND Kokkos_DEVICES "CUDA" _kokkos_cuda_found)
if (_kokkos_cuda_found GREATER -1)
  set(KR_CUDA_DEVICE_ENABLED TRUE)
  mark_as_advanced(KR_CUDA_DEVICE_ENABLED)
endif()
list(FIND Kokkos_DEVICES "OPENMP" _kokkos_omp_found)
if (_kokkos_omp_found GREATER -1)
  set(KR_OPENMP_DEVICE_ENABLED TRUE)
  mark_as_advanced(KR_OPENMP_DEVICE_ENABLED)
endif()

option(KR_ENABLE_EXEC_SPACES "enable resilient execution spaces" OFF)
cmake_dependent_option(KR_ENABLE_CUDA_EXEC_SPACE "enable the resilient CUDA execution space" ON "KR_ENABLE_EXEC_SPACES;KR_CUDA_DEVICE_ENABLED" OFF)
cmake_dependent_option(KR_ENABLE_OPENMP_EXEC_SPACE "enable the resilient CUDA execution space" ON "KR_ENABLE_EXEC_SPACES;KR_OPENMP_DEVICE_ENABLED" OFF)

if (KR_ENABLE_CUDA_EXEC_SPACE)
  message(STATUS "CUDA resilient execution spaces are enabled")
endif()

if (KR_ENABLE_OPENMP_EXEC_SPACE)
  message(STATUS "OpenMP resilient execution spaces are enabled")
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
  target_compile_definitions(resilience PUBLIC KR_ENABLE_CUDA)
endif()

if (KR_ENABLE_WRAPPER)
  target_compile_definitions(resilience PUBLIC KR_ENABLE_WRAPPER)
endif()

if (KR_ENABLE_TMR)
  target_compile_definitions(resilience PUBLIC KR_ENABLE_TMR)
endif()

if (KR_ENABLE_DMR)
  target_compile_definitions(resilience PUBLIC KR_ENABLE_DMR)
endif()

# Set up find_package config and version file
# Allow same major version compatibility
# Possibly in the future find_package(Kokkos REQUIRED COMPONENTS reslience)
include(CMakePackageConfigHelpers)
configure_package_config_file(cmake/resilienceConfig.cmake.in
                              ${CMAKE_CURRENT_BINARY_DIR}/resilienceConfig.cmake
                              INSTALL_DESTINATION ${CMAKE_INSTALL_FULL_LIBDIR}/cmake)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/resilienceConfigVersion.cmake
                                 COMPATIBILITY SameMajorVersion
                                 )

# Set install rules
install(TARGETS resilience EXPORT resilienceTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        )

install(EXPORT resilienceTargets
        FILE resilienceTargets.cmake
        NAMESPACE Kokkos::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/resilience
        )


install(FILES ${CMAKE_CURRENT_BINARY_DIR}/resilienceConfig.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/resilienceConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/resilience
        )

install(DIRECTORY src/resilience DESTINATION include FILES_MATCHING PATTERN "*.hpp")

if (KR_ENABLE_HDF5)
  install(FILES
          ${CMAKE_MODULE_PATH}/FindHDF5.cmake
          DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/resilience/Modules )

  # Need to install/export HDF5 find module for downstream dependencies
  configure_file(cmake/Modules/FindHDF5.cmake Modules/FindHDF5.cmake COPYONLY)
endif()

configure_file(src/resilience/config/Config.hpp.in ${CMAKE_CURRENT_BINARY_DIR}/src/resilience/config/Config.hpp @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/src/resilience/config/Config.hpp DESTINATION include/resilience/config)

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
