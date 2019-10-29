target_sources(resilience PRIVATE
               ${CMAKE_CURRENT_LIST_DIR}/Resilience.cpp
               ${CMAKE_CURRENT_LIST_DIR}/AutomaticCheckpoint.cpp
               ${CMAKE_CURRENT_LIST_DIR}/Context.cpp
               ${CMAKE_CURRENT_LIST_DIR}/Config.cpp
               ${CMAKE_CURRENT_LIST_DIR}/Cref.cpp
               )

add_subdirectory(filesystem)
add_subdirectory(stdio)

if (KR_ENABLE_VELOC)
  add_subdirectory(veloc)
endif()

if (KR_ENABLE_HDF5)
  add_subdirectory(hdf5)
endif()

if (Kokkos_ENABLE_Cuda)
  add_subdirectory(cuda)
endif()

