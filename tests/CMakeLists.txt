
find_package(GTest REQUIRED)

add_executable(resilience_tests
               TestMain.cpp
               TestResilience.cpp
               )
add_test(NAME resilience_tests COMMAND resilience_tests)

if (KR_ENABLE_VELOC)
   add_executable(veloc_test
                  TestVelocMemoryBackend.cpp
                  )
   add_test(NAME veloc_test COMMAND veloc_test)
   target_link_libraries(veloc_test PRIVATE resilience)
endif()

if (KR_ENABLE_STDIO)
  target_sources(resilience_tests PRIVATE
                 ${CMAKE_CURRENT_LIST_DIR}/TestViewCheckpoint.cpp
                 )
endif()

target_link_libraries(resilience_tests PRIVATE GTest::GTest)
target_link_libraries(resilience_tests PRIVATE resilience)

target_link_libraries(resilience_tests PUBLIC Boost::boost)
target_link_libraries(resilience_tests PRIVATE Boost::filesystem)

configure_file(data/hdf5_test_config.json data/hdf5_test_config.json COPYONLY)
configure_file(data/veloc_test.cfg data/veloc_test.cfg COPYONLY)
