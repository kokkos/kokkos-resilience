if (KR_ENABLE_STDFILE)
  target_sources(resilience PRIVATE
                 ${CMAKE_CURRENT_LIST_DIR}/StdFileBackend.cpp
                 )
endif()