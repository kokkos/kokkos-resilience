
include(FetchContent)
FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG 10b1902d893ea8cc43c69541d70868f91af3646b
    )

FetchContent_MakeAvailable(googletest)

add_executable(resilience_tests
               TestMain.cpp
               TestResilience.cpp
               )

if (KR_ENABLE_VELOC)
  target_sources(resilience_tests PRIVATE
      TestVelocMemoryBackend.cpp)
endif()

if (KR_ENABLE_STDIO)
  target_sources(resilience_tests PRIVATE
                 TestViewCheckpoint.cpp
                 )
endif()


target_link_libraries(resilience_tests PRIVATE gtest)
target_link_libraries(resilience_tests PRIVATE resilience)

include(GoogleTest)
gtest_discover_tests(resilience_tests)

configure_file(data/hdf5_test_config.json data/hdf5_test_config.json COPYONLY)
configure_file(data/veloc_test.cfg data/veloc_test.cfg COPYONLY)
