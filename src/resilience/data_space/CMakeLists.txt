if (KR_ENABLE_STDFILE_DATA_SPACE)
  target_sources(resilience PRIVATE ${CMAKE_CURRENT_LIST_DIR}/StdFileSpace.cpp)
endif()

if (KR_ENABLE_HDF5_DATA_SPACE)
  target_sources(resilience PRIVATE ${CMAKE_CURRENT_LIST_DIR}/HDF5Space.cpp)

  install(FILES
          ${CMAKE_MODULE_PATH}/FindHDF5.cmake
          DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/resilience/Modules)

  # Need to install/export HDF5 find module for downstream dependencies
  configure_file(${CMAKE_LIST_DIR}/cmake/Modules/FindHDF5.cmake Modules/FindHDF5.cmake COPYONLY)
endif()
