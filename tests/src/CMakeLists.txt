
#
# Add test-only library for gtest to be reused by all the subpackages
#
SET(GTEST_SOURCE_DIR ${PROJECT_SOURCE_DIR}/tpl/gtest)
SET(RESILIENCE_TEST_SRC_DIR ${PROJECT_SOURCE_DIR}/tests/src)

MACRO(RESILIENCE_ADD_TEST EXE_NAME)
  CMAKE_PARSE_ARGUMENTS(PARSE 
    ""
    ""
    "SOURCES"
    ${ARGN})
  ADD_EXECUTABLE(${EXE_NAME} ${PARSE_SOURCES})
  TARGET_LINK_KOKKOS(${EXE_NAME} PRIVATE)
  TARGET_LINK_LIBRARIES(${EXE_NAME} PRIVATE kokkos_gtest)
  TARGET_LINK_LIBRARIES(${EXE_NAME} PRIVATE resilience)
  TARGET_INCLUDE_DIRECTORIES(${EXE_NAME} PRIVATE ${RESILIENCE_TEST_SRC_DIR})
  TARGET_INCLUDE_DIRECTORIES(${EXE_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/src/resilience)
ENDMACRO(RESILIENCE_ADD_TEST)

INCLUDE_DIRECTORIES(${GTEST_SOURCE_DIR}/gtest)
INCLUDE_DIRECTORIES(${RESILIENCE_TEST_SRC_DIR})

ADD_LIBRARY(kokkos_gtest ${GTEST_SOURCE_DIR}/gtest/gtest-all.cc)
target_compile_options(kokkos_gtest
                       PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${KOKKOS_CXX_FLAGS}>
                       )
TARGET_LINK_KOKKOS(kokkos_gtest PRIVATE)
  
TARGET_COMPILE_DEFINITIONS(kokkos_gtest PUBLIC "-DGTEST_HAS_PTHREAD=0")
TARGET_INCLUDE_DIRECTORIES(kokkos_gtest PUBLIC ${GTEST_SOURCE_DIR})


IF(Kokkos_ENABLE_OpenMP)
   RESILIENCE_ADD_TEST(Kokkos_TestOpenMP_Resilience
                       SOURCES 
                       ${RESILIENCE_TEST_SRC_DIR}/UnitTestMainInit.cpp 
                       ${RESILIENCE_TEST_SRC_DIR}/openmp/TestOpenMP_Resilience.cpp
                      )

ENDIF()

IF(Kokkos_ENABLE_Cuda)
   RESILIENCE_ADD_TEST(Kokkos_TestCuda_Resilience
                       SOURCES 
                       ${RESILIENCE_TEST_SRC_DIR}/UnitTestMainInit.cpp 
                       ${RESILIENCE_TEST_SRC_DIR}/cuda/TestCuda_Resilience.cpp
                      )

ENDIF()
#
# Define the tests
#
