target_sources(resilience PRIVATE
               ${CMAKE_CURRENT_LIST_DIR}/Resilience.cpp
               ${CMAKE_CURRENT_LIST_DIR}/Config.cpp
               ${CMAKE_CURRENT_LIST_DIR}/ResilientRef.cpp
               )

add_subdirectory(registration)
add_subdirectory(context)
add_subdirectory(backend)

add_subdirectory(view_hooks)

add_subdirectory(stdio)

if (KR_ENABLE_CUDA_EXEC_SPACE)
  add_subdirectory(cuda)
endif()

if (KR_ENABLE_OPENMP_EXEC_SPACE)
  add_subdirectory(openMP)
endif()

