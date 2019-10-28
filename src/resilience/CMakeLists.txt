target_sources(resilience PRIVATE
               ${CMAKE_CURRENT_LIST_DIR}/Resilience.cpp
               )

add_subdirectory(impl)
add_subdirectory(filesystem)
add_subdirectory(stdio)

if (KR_ENABLE_VELOC)
  add_subdirectory(veloc)
endif()

if (KR_ENABLE_HDF5)
  add_subdirectory(hdf5)
endif()

if (Kokkos_ENABLE_CUDA)
  add_subdirectory(cuda)
endif()

