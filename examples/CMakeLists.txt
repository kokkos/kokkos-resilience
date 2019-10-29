macro(add_example _target)
  message(STATUS "example args: ${ARGN}")
  add_executable(${_target} ${ARGN})
  target_link_libraries(${_target} PRIVATE Kokkos::resilience)
  set_property(TARGET ${_target} PROPERTY CXX_STANDARD 14)
endmacro()

if (KR_ENABLE_VELOC)
  add_custom_command(OUTPUT config.json
      COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/config.json" "${CMAKE_CURRENT_BINARY_DIR}/config.json"
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/config.json)

   add_example(simple_checkpoint SimpleCheckpoint.cpp "${CMAKE_CURRENT_BINARY_DIR}/config.json")

   add_custom_command(TARGET simple_checkpoint POST_BUILD
       COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/veloc_test.cfg ${CMAKE_CURRENT_BINARY_DIR}/veloc_test.cfg)
endif()
