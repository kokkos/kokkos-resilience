
find_package(GTest REQUIRED)

add_executable(resilience_tests
               TestMain.cpp
               TestViewCheckpoint.cpp
               TestResilience.cpp
               TestVelocMemoryBackend.cpp
               )


target_link_libraries(resilience_tests PRIVATE GTest::GTest)
target_link_libraries(resilience_tests PRIVATE resilience)

configure_file(data/hdf5_test_config.json data/hdf5_test_config.json COPYONLY)
configure_file(data/veloc_test.cfg data/veloc_test.cfg COPYONLY)