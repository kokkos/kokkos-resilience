
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
  target_link_libraries(${_target} PRIVATE cxxopts::cxxopts)
  if (KR_ENABLE_MPI_BACKENDS)
     target_link_libraries(${_target} PRIVATE MPI::MPI_CXX)
  endif()
  if (KR_ENABLE_MAGISTRATE)
     target_link_libraries(${_target} PRIVATE vt::lib::checkpoint)
  endif()
  if (KR_ENABLE_VT)
     target_link_libraries(${_target} PRIVATE vt::runtime::vt)
  endif()

  if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(${_target} PRIVATE "-Werror" "-Wall")
  endif()
endfunction()

if (KR_ENABLE_VELOC)
    add_example(simple_checkpoint SOURCES SimpleCheckpoint.cpp
       RESOURCES config.json veloc_test.cfg)
    add_example(benchmark_multiview SOURCES benchmark_multiviews.cpp
       RESOURCES config.json veloc_test.cfg)
    add_example(ping_pong SOURCES ping_pong.cpp RESOURCES config.json veloc_test.cfg)
endif()

if (KR_ENABLE_STDFILE)
    add_example(simple_file_checkpoint SOURCES SimpleFileCheckpoint.cpp
        RESOURCES config_file.json file_test.cfg)
endif()

if(KR_ENABLE_VT)
    add_example(jacobi_checkpoint SOURCES jacobi/main.cpp jacobi/solver.cpp jacobi/config.cpp
        RESOURCES jacobi/config_jacobi.json jacobi/config_jacobi_async.json jacobi/config_jacobi_1.json jacobi/config_jacobi_more_async.json)
endif()
