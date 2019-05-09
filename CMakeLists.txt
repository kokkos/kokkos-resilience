cmake_minimum_required(VERSION 3.13)
project(kokkos-resilience VERSION 0.1.0)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")

add_library(resilience)
add_library(Kokkos::resilience ALIAS resilience)

add_subdirectory(src)

# Require C++14
target_compile_features(resilience PUBLIC cxx_std_14)

target_include_directories(resilience PUBLIC
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                           $<INSTALL_INTERFACE:include>
                           )

# TODO: change this when Kokkos cmake gets updated
find_package(Kokkos REQUIRED NO_CMAKE_PACKAGE_REGISTRY)
target_include_directories(kokkos INTERFACE ${Kokkos_INCLUDE_DIRS})
message("Kokkos include dirs: ${Kokkos_INCLUDE_DIRS}")

target_link_libraries(resilience PUBLIC kokkos)

# VeloC backend
find_package(VeloC REQUIRED)
target_link_libraries(resilience PRIVATE veloc::veloc)

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
                              INSTALL_DESTINATION cmake)
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

install(DIRECTORY include/resilience DESTINATION include FILES_MATCHING PATTERN "*.hpp")

# Add subdirectories for examples and tests if they are enabled
option(KR_ENABLE_TESTS "Enable tests in the build" ON)
option(KR_ENABLE_EXAMPLES "Enable examples in the build" ON)

if (KR_ENABLE_TESTS)
  add_subdirectory(tests)
endif()

if (KR_ENABLE_EXAMPLES)
  add_subdirectory(examples)
endif()