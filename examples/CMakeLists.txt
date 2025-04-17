
include(utility)
include(CMakePrintHelpers)

function(add_example _target)
  cmake_parse_arguments(ARG
      ""
      ""
      "RESOURCES;SOURCES"
      ${ARGN})

  add_executable(${_target} ${ARG_SOURCES})
  target_resources(${_target} PRIVATE ${ARG_RESOURCES})
  target_link_libraries(${_target} PRIVATE Kokkos::resilience)
  target_link_libraries(${_target} PRIVATE CLI11::CLI11)
  if (KR_ENABLE_VELOC OR KR_ENABLE_HDF5_PARALLEL)
     target_link_libraries(${_target} PRIVATE MPI::MPI_CXX)
  endif()

  if (KR_WARNINGS_AS_ERRORS)
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
      target_compile_options(${_target} PRIVATE "-Wall")
    endif()
  endif()

  if (KR_WARNINGS_AS_ERRORS)
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
      target_compile_options(${_target} PRIVATE "-Werror")
    endif()
  endif()
endfunction()

if (KR_ENABLE_VELOC)
    add_example(simple_checkpoint SOURCES SimpleCheckpoint.cpp
       RESOURCES config.json veloc_test.cfg)
    add_example(benchmark_multiview SOURCES benchmark_multiviews.cpp
       RESOURCES config.json veloc_test.cfg)
    add_example(ping_pong SOURCES ping_pong.cpp RESOURCES config.json veloc_test.cfg)
endif()

add_example(simple_file_checkpoint SOURCES SimpleFileCheckpoint.cpp
    RESOURCES config_file.json file_test.cfg)
