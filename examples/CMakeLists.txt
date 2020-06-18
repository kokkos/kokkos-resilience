
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
  if (KR_ENABLE_VELOC OR KR_ENABLE_HDF5_PARALLEL)
     target_link_libraries(${_target} PRIVATE MPI::MPI_CXX)
  endif()
endfunction()

if (KR_ENABLE_VELOC)
   add_example(simple_checkpoint SOURCES SimpleCheckpoint.cpp
       RESOURCES config.json veloc_test.cfg)
endif()

if (KR_ENABLE_STDFILE)
    add_example(simple_file_checkpoint SOURCES SimpleFileCheckpoint.cpp
        RESOURCES config_file.json file_test.cfg)
endif()
