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

if (KR_CUDA_EXEC_SPACE)
  add_subdirectory(cuda)
endif()

if (KR_OPENMP_EXEC_SPACE)
  add_subdirectory(openMP)
endif()

