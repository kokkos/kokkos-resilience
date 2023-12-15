target_sources(resilience PRIVATE ${CMAKE_CURRENT_LIST_DIR}/Automatic.cpp)

if (KR_ENABLE_VELOC)
  add_subdirectory(veloc)
endif()

add_subdirectory(stdfile)

if (KR_ENABLE_HDF5)
  add_subdirectory(hdf5)
endif()

add_subdirectory(filesystem)
