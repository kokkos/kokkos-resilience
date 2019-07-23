
find_package(GTest REQUIRED)

add_executable(resilience_tests
               test_main.cpp
               test_view_checkpoint.cpp
               test_resilience.cpp
               )


target_link_libraries(resilience_tests PRIVATE GTest::GTest)
target_link_libraries(resilience_tests PRIVATE resilience)

configure_file(data/hdf5_test_config.json data/hdf5_test_config.json COPYONLY)