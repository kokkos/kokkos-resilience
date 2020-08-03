
include(FetchContent)
FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG 10b1902d893ea8cc43c69541d70868f91af3646b
    )

FetchContent_MakeAvailable(googletest)

add_executable(resilience_tests
               TestMain.cpp
               TestLambdaCapture.cpp
               )
add_test(NAME resilience_tests COMMAND resilience_tests)
include(GoogleTest)

if (KR_ENABLE_VELOC)
   add_executable(veloc_test
                  TestMain.cpp
                  TestVelocMemoryBackend.cpp
                  )
   add_test(NAME veloc_test COMMAND veloc_test)
   target_link_libraries(veloc_test PRIVATE gtest)
   target_link_libraries(veloc_test PRIVATE resilience)
endif()

if (KR_ENABLE_VELOC)
  target_sources(resilience_tests PRIVATE
      TestVelocMemoryBackend.cpp)
  target_link_libraries(resilience_tests PRIVATE MPI::MPI_CXX)
endif()


if (KR_ENABLE_STDFILE)
  target_sources(resilience_tests PRIVATE
      TestStdFileBackend.cpp)
endif()

if (KR_ENABLE_STDIO)
  target_sources(resilience_tests PRIVATE
                 TestViewCheckpoint.cpp
                 )
endif()

  target_sources(resilience_tests PRIVATE
                 TestResilience.cpp
                )
gtest_discover_tests(resilience_tests)

target_link_libraries(resilience_tests PRIVATE gtest)
target_link_libraries(resilience_tests PRIVATE resilience)
##target_link_libraries(resilience_tests PUBLIC Boost::boost)
##target_link_libraries(resilience_tests PRIVATE Boost::filesystem)

configure_file(data/hdf5_test_config.json data/hdf5_test_config.json COPYONLY)
configure_file(data/veloc_test.cfg data/veloc_test.cfg COPYONLY)
