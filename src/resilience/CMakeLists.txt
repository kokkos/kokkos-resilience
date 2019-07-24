target_sources(resilience PRIVATE
               ${CMAKE_CURRENT_LIST_DIR}/Kokkos_Resilience.cpp
               )

add_subdirectory(filesystem)
add_subdirectory(stdio)

if (USE_VELOC)
  add_subdirectory(veloc)
endif()

if (USE_HDF5)
  add_subdirectory(hdf5)
endif()

if (Kokkos_ENABLE_Cuda)
  add_subdirectory(cuda)
endif()

