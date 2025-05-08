cmake_minimum_required(VERSION 3.17)
project(kokkos-resilience VERSION 0.1.0)

include(CMakeDependentOption)

INCLUDE(GNUInstallDirs)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")

add_library(resilience)
add_library(Kokkos::resilience ALIAS resilience)

#Expose a variable to linking CMakeLists and as a compile definition.
set(KR_EXPOSED_OPTIONS "")
macro(KR_EXPOSE_OPTION OPT)
  list(APPEND KR_EXPOSED_OPTIONS "${OPT}")

  if (${OPT})
    target_compile_definitions(resilience PUBLIC ${OPT})
  endif()
endmacro()

#Helper for making and exposing a basic option with optional dependencies
#If dependencies are listed, this option is false unless dependencies are met
macro(KR_OPTION OPT DESC DEFAULT)
  set(DEPS "${ARGN}") #DEPS gets all extra arguments
  if (DEPS)
    cmake_dependent_option(${OPT} "${DESC}" ${DEFAULT} "${DEPS}" OFF)
  else()
    option(${OPT} "${DESC}" ${DEFAULT})
  endif()
  KR_EXPOSE_OPTION(${OPT})
endmacro()


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
find_package(Boost REQUIRED)

set_property(TARGET resilience PROPERTY CXX_STANDARD ${Kokkos_CXX_STANDARD})
target_link_libraries(resilience PUBLIC Kokkos::kokkos Boost::boost)

#Make individual variables for available Kokkos devices
foreach(DEVICE ${Kokkos_DEVICES})
  set(KR_${DEVICE}_DEVICE_ENABLED TRUE)
  mark_as_advanced(KR_${DEVICE}_DEVICE_ENABLED)
endforeach()

#Library options
kr_option(KR_ENABLE_TRACING "Enable tracing of resilience functions" OFF)
option(KR_ENABLE_TESTS "Enable tests in the build" ON)
option(KR_ENABLE_EXAMPLES "Enable examples in the build" ON)

#Automatic checkpointing options
kr_option(KR_ENABLE_AUTOMATIC_CHECKPOINTING "Compile automatic checkpointing contexts and backends" ON)
#Consistency contexts
kr_option(KR_ENABLE_MPI_CONTEXT "Compile MPI checkpointing context" ON KR_ENABLE_AUTOMATIC_CHECKPOINTING)
#Backends
kr_option(KR_ENABLE_VELOC_BACKEND "Compile VeloC checkpointing backend" ON KR_ENABLE_MPI_CONTEXT)
kr_option(KR_VELOC_BAREBONE "Use the barebone branch of VeloC" OFF KR_ENABLE_VELOC)

#Data space options
kr_option(KR_ENABLE_DATA_SPACES "Enable Kokkos memory spaces for manual view checkpointing" OFF)
kr_option(KR_ENABLE_STDFILE_DATA_SPACE "Enable stdfile-based data space" ON KR_ENABLE_DATA_SPACES)
kr_option(KR_ENABLE_HDF5_DATA_SPACE "Enable HDF5-based data space" OFF KR_ENABLE_DATA_SPACES)
kr_option(KR_ENABLE_HDF5_PARALLEL "use parallel version of HDF5" OFF KR_ENABLE_HDF5_DATA_SPACE)

#Exec space options
kr_option(KR_ENABLE_EXEC_SPACES "enable resilient execution spaces" OFF)
kr_option(KR_ENABLE_OPENMP_EXEC_SPACE "enable the resilient OpenMP execution space" ON "KR_ENABLE_EXEC_SPACES;KR_OPENMP_DEVICE_ENABLED")
kr_option(KR_ENABLE_DMR "enable double modular redundancy" OFF "KR_ENABLE_OPENMP_EXEC_SPACE")
kr_option(KR_ENABLE_TMR "enable triple modular redundancy" ON "KR_ENABLE_OPENMP_EXEC_SPACE;NOT KR_ENABLE_DMR")
kr_option(KR_ENABLE_WRAPPER "enable kernel fusing" OFF "KR_ENABLE_OPENMP_EXEC_SPACE")

# VeloC backend
if (KR_ENABLE_VELOC_BACKEND)
  find_package(veloc REQUIRED)
  target_link_libraries(resilience PUBLIC veloc::client)
endif()

if (KR_ENABLE_HDF5_DATA_SPACE)
   find_package(HDF5 REQUIRED)
   target_link_libraries(resilience PUBLIC HDF5::HDF5)
endif()

# MPI requirement
if (KR_ENABLE_MPI_CONTEXT OR KR_ENABLE_HDF5_PARALLEL)
   find_package(MPI REQUIRED)
   target_link_libraries(resilience PRIVATE MPI::MPI_CXX)
endif()

if (KR_ENABLE_OPENMP_EXEC_SPACE)
  message(STATUS "OpenMP resilient execution spaces are enabled")
endif()


add_subdirectory(src)


#Set variable values to expose to resilienceConfig.cmake
set(KR_EXPOSED_OPTION_VALUES "")
foreach (OPT ${KR_EXPOSED_OPTIONS})
  list(APPEND KR_EXPOSED_OPTION_VALUES ${${OPT}})
endforeach()


# Export targets for in-tree linking
export(TARGETS resilience
       NAMESPACE Kokkos::
       FILE resilienceTargets.cmake
       )

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


# Add subdirectories for examples and tests if they are enabled
add_subdirectory(tpl)

if (KR_ENABLE_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if (KR_ENABLE_EXAMPLES)
  add_subdirectory(examples)
endif()
