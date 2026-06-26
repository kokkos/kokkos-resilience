target_sources(resilience PRIVATE ${CMAKE_CURRENT_LIST_DIR}/StdFileBackend.cpp)

if (KR_ENABLE_VELOC_BACKEND)
  target_sources(resilience PRIVATE ${CMAKE_CURRENT_LIST_DIR}/VelocBackend.cpp)
endif()

if (KR_ENABLE_FENIX_BACKEND)
  target_sources(resilience PRIVATE ${CMAKE_CURRENT_LIST_DIR}/FenixBackend.cpp)
endif()
