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

list(FIND Kokkos_DEVICES "CUDA" _cuda_found)

if (${_cuda_found} GREATER "-1")
  add_subdirectory(cuda)
endif()

list(FIND Kokkos_DEVICES "OPENMP" _omp_found)

if (${_omp_found} GREATER "-1")
  add_subdirectory(openMP)
endif()

