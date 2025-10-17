target_sources(resilience PRIVATE ${CMAKE_CURRENT_LIST_DIR}/StdFile.cpp
                                  ${CMAKE_CURRENT_LIST_DIR}/Backend.cpp)

if (KR_ENABLE_VELOC_BACKEND)
  target_sources(resilience PRIVATE ${CMAKE_CURRENT_LIST_DIR}/Veloc.cpp)
endif()
