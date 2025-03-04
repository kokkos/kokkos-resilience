target_sources(resilience PRIVATE ${CMAKE_CURRENT_LIST_DIR}/Resilience.cpp)

if (KR_ENABLE_AUTOMATIC_CHECKPOINTING)
  target_sources(resilience PRIVATE 
                 ${CMAKE_CURRENT_LIST_DIR}/config/Config.cpp
                 ${CMAKE_CURRENT_LIST_DIR}/Cref.cpp
                 )
  add_subdirectory(backend)
  add_subdirectory(context)
endif()

if (KR_ENABLE_EXEC_SPACES)
  add_subdirectory(exec_space)
endif()

if (KR_ENABLE_AUTOMATIC_CHECKPOINTING OR KR_ENABLE_EXEC_SPACES)
  add_subdirectory(view_hooks)
endif()

if (KR_ENABLE_DATA_SPACES)
  add_subdirectory(data_space)
endif()

add_subdirectory(util)
