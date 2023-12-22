
include(FetchContent)
FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.11.0
    )

FetchContent_MakeAvailable(googletest)

add_executable(resilience_tests
               TestMain.cpp
#              TestResilience.cpp
#               TestLambdaCapture.cpp
#               TestHooks.cpp
	        TestOpenMPResilientExecution.cpp
		TestMiniMD.cpp
#               TestDynamicViewHooks.cpp
               )

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

if (KR_ENABLE_HDF5)
  target_sources(resilience_tests PRIVATE
                 TestHDF5Configuration.cpp
                 )
endif()

if (KR_ENABLE_OPENMP_EXEC_SPACE)
  target_sources(resilience_tests PRIVATE
                 TestOpenMPResilientExecution.cpp
                 )
endif()

if (KR_WARNINGS_AS_ERRORS)
  if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(resilience_tests PRIVATE "-Wall")
  endif()
endif()

if (KR_WARNINGS_AS_ERRORS)
  if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(resilience_tests PRIVATE "-Werror")
  endif()
endif()

target_link_libraries(resilience_tests PRIVATE gtest)
target_link_libraries(resilience_tests PRIVATE resilience)

include(GoogleTest)
gtest_discover_tests(resilience_tests)

set(KR_TEST_DATADIR "${CMAKE_CURRENT_BINARY_DIR}/data")

configure_file(data/hdf5_test_config.json data/hdf5_test_config.json COPYONLY)
configure_file(data/veloc_test.cfg.in data/veloc_test.cfg @ONLY)

target_compile_definitions(resilience_tests PRIVATE KR_TEST_DATADIR="${KR_TEST_DATADIR}")
