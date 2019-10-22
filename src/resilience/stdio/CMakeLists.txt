if (KR_ENABLE_STDIO)
  target_sources(resilience PRIVATE
                 ${CMAKE_CURRENT_LIST_DIR}/StdFileSpace.cpp
                 )
endif()