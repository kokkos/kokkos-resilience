
include(FetchContent)
FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.16.0
    )

FetchContent_MakeAvailable(googletest)

add_executable(resilience_tests
               TestMain.cpp
               TestHooks.cpp
               TestDynamicViewHooks.cpp
               )

if (KR_ENABLE_VELOC_BACKEND)
  target_sources(resilience_tests PRIVATE
  		TestVelocMemoryBackend.cpp)
  target_link_libraries(resilience_tests PRIVATE MPI::MPI_CXX)
endif()

if (KR_ENABLE_AUTOMATIC_CHECKPOINTING)
  target_sources(resilience_tests PRIVATE
     		 TestStdFileBackend.cpp
                 TestLambdaCapture.cpp
		 )
endif()

if (KR_ENABLE_STDFILE_DATA_SPACE)
  target_sources(resilience_tests PRIVATE
                 TestViewCheckpoint.cpp
                 )
endif()

if (KR_ENABLE_HDF5_DATA_SPACE)
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
target_compile_definitions(resilience_tests PRIVATE KR_TEST_DATADIR="${KR_TEST_DATADIR}")

configure_file(data/hdf5_test_config.json data/hdf5_test_config.json COPYONLY)
configure_file(data/veloc_test.cfg.in data/veloc_test.cfg @ONLY)

# fenix tests

if (KR_ENABLE_FENIX_BACKEND)
  add_executable(resilience_fenix_tests
                 TestFenixMemoryBackend.cpp
                 )
  target_link_libraries(resilience_fenix_tests PRIVATE gtest)
  target_link_libraries(resilience_fenix_tests PRIVATE resilience)
  target_link_libraries(resilience_fenix_tests PRIVATE MPI::MPI_CXX)

  if (KR_WARNINGS_AS_ERRORS)
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
      target_compile_options(resilience_fenix_tests PRIVATE "-Wall")
    endif()
  endif()

  if (KR_WARNINGS_AS_ERRORS)
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
      target_compile_options(resilience_fenix_tests PRIVATE "-Werror")
    endif()
  endif()

  if(KR_SERIAL_DEVICE_ENABLED)
    add_test(
      NAME "TestFenixMemoryBackend.Serial.np1"
      COMMAND ${MPIEXEC_EXECUTABLE} -np 1 --with-ft mpi $<TARGET_FILE:resilience_fenix_tests> --gtest_filter="TestFenixMemoryBackend.Serial")
    add_test(
      NAME "TestFenixMemoryBackend.Serial.np2"
      COMMAND ${MPIEXEC_EXECUTABLE} -np 2 --with-ft mpi $<TARGET_FILE:resilience_fenix_tests> --gtest_filter="TestFenixMemoryBackend.Serial")
    add_test(
      NAME "TestFenixMemoryBackend.Serial.np3.prockill"
      COMMAND ${MPIEXEC_EXECUTABLE} -np 3 --with-ft mpi $<TARGET_FILE:resilience_fenix_tests> --gtest_filter="TestFenixMemoryBackend.Serial")
    add_test(
      NAME "TestFenixMemoryBackend.Serial.np4.prockill"
      COMMAND ${MPIEXEC_EXECUTABLE} -np 4 --with-ft mpi $<TARGET_FILE:resilience_fenix_tests> --gtest_filter="TestFenixMemoryBackend.Serial")
  endif()

  if(KR_OPENMP_DEVICE_ENABLED)
    add_test(
      NAME "TestFenixMemoryBackend.OpenMP.np1"
      COMMAND ${MPIEXEC_EXECUTABLE} -np 1 --with-ft mpi $<TARGET_FILE:resilience_fenix_tests> --gtest_filter="TestFenixMemoryBackend.OpenMP")
    add_test(
      NAME "TestFenixMemoryBackend.OpenMP.np2"
      COMMAND ${MPIEXEC_EXECUTABLE} -np 2 --with-ft mpi $<TARGET_FILE:resilience_fenix_tests> --gtest_filter="TestFenixMemoryBackend.OpenMP")
    add_test(
      NAME "TestFenixMemoryBackend.OpenMP.np3.prockill"
      COMMAND ${MPIEXEC_EXECUTABLE} -np 3 --with-ft mpi $<TARGET_FILE:resilience_fenix_tests> --gtest_filter="TestFenixMemoryBackend.OpenMP")
    add_test(
      NAME "TestFenixMemoryBackend.OpenMP.np4.prockill"
      COMMAND ${MPIEXEC_EXECUTABLE} -np 4 --with-ft mpi $<TARGET_FILE:resilience_fenix_tests> --gtest_filter="TestFenixMemoryBackend.OpenMP")
  endif()
endif()
