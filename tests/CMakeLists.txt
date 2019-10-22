
find_package(GTest REQUIRED)

add_executable(resilience_tests
               TestMain.cpp
               TestResilience.cpp
               TestVelocMemoryBackend.cpp
               )

if (KR_ENABLE_STDIO)
  target_sources(resilience_tests PRIVATE
                 ${CMAKE_CURRENT_LIST_DIR}/TestViewCheckpoint.cpp
                 )
endif()


target_link_libraries(resilience_tests PRIVATE GTest::GTest)
target_link_libraries(resilience_tests PRIVATE resilience)

configure_file(data/hdf5_test_config.json data/hdf5_test_config.json COPYONLY)
configure_file(data/veloc_test.cfg data/veloc_test.cfg COPYONLY)