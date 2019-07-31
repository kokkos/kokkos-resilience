macro(add_example _target)
  add_executable(${_target} ${ARGN})
  target_link_libraries(${_target} PRIVATE Kokkos::resilience)
endmacro()

if (USE_VELOC)
   add_example(simple_checkpoint SimpleCheckpoint.cpp)
   add_custom_command(TARGET simple_checkpoint PRE_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/veloc_test.cfg ${CMAKE_CURRENT_BINARY_DIR}/veloc_test.cfg)
endif()
