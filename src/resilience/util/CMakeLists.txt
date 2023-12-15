target_sources(resilience PRIVATE
               ${CMAKE_CURRENT_LIST_DIR}/Trace.cpp
               )

if(KR_ENABLE_VT)
  target_sources(resilience PRIVATE
                ${CMAKE_CURRENT_LIST_DIR}/VTUtil.cpp
                )
endif()
