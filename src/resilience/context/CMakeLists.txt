target_sources(resilience PRIVATE ${CMAKE_CURRENT_LIST_DIR}/Context.cpp)

if (KR_ENABLE_MPI_BACKENDS)
  target_sources(resilience PRIVATE ${CMAKE_CURRENT_LIST_DIR}/MPIContext.cpp)
endif()

if (KR_ENABLE_STDFILE)
  target_sources(resilience PRIVATE ${CMAKE_CURRENT_LIST_DIR}/StdFileContext.cpp)
endif()
