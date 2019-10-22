
find_package(GTest REQUIRED)

add_executable(resilience_tests
               TestMain.cpp
               TestViewCheckpoint.cpp
               TestResilience.cpp
               )
add_test(NAME resilience_tests COMMAND resilience_tests)

if (KR_ENABLE_VELOC)
   add_executable(veloc_test
                  TestVelocMemoryBackend.cpp
                  )
   add_test(NAME veloc_test COMMAND veloc_test)
endif()


target_link_libraries(resilience_tests PRIVATE GTest::GTest)
target_link_libraries(resilience_tests PRIVATE resilience)

configure_file(data/hdf5_test_config.json data/hdf5_test_config.json COPYONLY)
