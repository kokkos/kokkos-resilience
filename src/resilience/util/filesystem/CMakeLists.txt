target_sources(resilience PRIVATE
               ${CMAKE_CURRENT_LIST_DIR}/Filesystem.cpp
               )

if (KR_ENABLE_DATA_SPACES)
    target_sources(resilience PRIVATE
                   ${CMAKE_CURRENT_LIST_DIR}/ExternalIOInterface.cpp
                   )
endif()
