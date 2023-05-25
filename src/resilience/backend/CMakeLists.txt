target_sources(resilience PRIVATE ${CMAKE_CURRENT_LIST_DIR}/Automatic.cpp)

if (KR_ENABLE_VELOC)
  add_subdirectory(veloc)
endif()

if (KR_ENABLE_STDFILE)
  add_subdirectory(stdfile)
endif()

if (KR_ENABLE_HDF5)
  add_subdirectory(hdf5)
endif()

add_subdirectory(filesystem)
