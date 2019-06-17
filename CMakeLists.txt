cmake_minimum_required(VERSION 3.12)
project(kokkos-resilience VERSION 0.1.0)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")

add_library(resilience)
add_library(Kokkos::resilience ALIAS resilience)

include_directories(${kokkos-resilience_SOURCE_DIR})
add_subdirectory(src)

# Require C++14
target_compile_features(resilience PUBLIC cxx_std_14)

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

# VeloC backend
find_package(VeloC REQUIRED)
target_link_libraries(resilience PRIVATE veloc::veloc)

# MPI requirement
find_package(MPI REQUIRED)
target_link_libraries(resilience PRIVATE MPI::MPI_CXX)

# Library options
option(KR_ENABLE_TRACING "Enable tracing of resilience functions" OFF)
if (KR_ENABLE_TRACING)
  target_compile_definitions(resilience PUBLIC KR_ENABLE_TRACING)
endif()

option( USE_HDF5 "add HDF5 support" ON )
option( USE_HDF5_PARALLEL "use parallel version of HDF5" ON )

if (USE_HDF5_PARALLEL)
  set(USE_HDF5 ON)
endif()

if (USE_HDF5)
   find_package(HDF5 REQUIRED)
   target_link_libraries(resilience PRIVATE HDF5::HDF5)
endif()

link_directories(${VeloC_LINK_DIRECTORIES}
                 ${Kokkos_LINK_DIRECTORIES} )

##link_libraries(dl)
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


install(FILES 
        ${CMAKE_MODULE_PATH}/FindVeloC.cmake
        ${CMAKE_MODULE_PATH}/FindHDF5.cmake
        DESTINATION ${CMAKE_INSTALL_PREFIX}/cmake/Modules )

# Need to install/export VeloC find module for downstream dependencies
configure_file(cmake/Modules/FindVeloC.cmake Modules/FindVeloC.cmake COPYONLY)

# Add subdirectories for examples and tests if they are enabled
option(KR_ENABLE_TESTS "Enable tests in the build" ON)
option(KR_ENABLE_EXAMPLES "Enable examples in the build" ON)

add_subdirectory(tpl)

if (KR_ENABLE_TESTS)
  add_subdirectory(tests)
endif()

if (KR_ENABLE_EXAMPLES)
  add_subdirectory(examples)
endif()

##get_cmake_property(_variableNames VARIABLES)
##list (SORT _variableNames)
##foreach (_variableName ${_variableNames})
##    message(STATUS "${_variableName}=${${_variableName}}")
##endforeach()
