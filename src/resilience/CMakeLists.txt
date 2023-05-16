target_sources(resilience PRIVATE
               ${CMAKE_CURRENT_LIST_DIR}/Resilience.cpp
               ${CMAKE_CURRENT_LIST_DIR}/Config.cpp
               ${CMAKE_CURRENT_LIST_DIR}/ResilientRef.cpp
               )

add_subdirectory(context)
add_subdirectory(registration)
add_subdirectory(filesystem)
add_subdirectory(stdio)

if (KR_ENABLE_VELOC)
  add_subdirectory(veloc)
endif()

if (KR_ENABLE_STDFILE)
  add_subdirectory(stdfile)
endif()

if (KR_ENABLE_HDF5)
  add_subdirectory(hdf5)
endif()

add_subdirectory(view_hooks)

list(FIND Kokkos_DEVICES "CUDA" _cuda_found)

if (${_cuda_found} GREATER "-1")
  add_subdirectory(cuda)
endif()

list(FIND Kokkos_DEVICES "OPENMP" _omp_found)

if (${_omp_found} GREATER "-1")
  add_subdirectory(openMP)
endif()

