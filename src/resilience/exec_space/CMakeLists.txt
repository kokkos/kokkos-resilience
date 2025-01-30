if (KR_ENABLE_CUDA_EXEC_SPACE)
  add_subdirectory(cuda)
endif()

if (KR_ENABLE_OPENMP_EXEC_SPACE)
  add_subdirectory(openMP)
endif()
