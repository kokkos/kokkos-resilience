target_sources(resilience PRIVATE
               ${PROJECT_SOURCE_DIR}/src/resilience/Kokkos_Resilience.cpp
               ${PROJECT_SOURCE_DIR}/src/resilience/util/trace.cpp
               ${PROJECT_SOURCE_DIR}/src/resilience/impl/Kokkos_ExternalIOInterface.cpp
               ${PROJECT_SOURCE_DIR}/src/resilience/impl/Kokkos_StdFileSpace.cpp
               )

if (USE_VELOC)
   target_sources(resilience PRIVATE
               ${PROJECT_SOURCE_DIR}/src/resilience/veloc/veloc_backend.cpp)
endif()

INSTALL (DIRECTORY
         "${PROJECT_SOURCE_DIR}/src/resilience/"
         DESTINATION ${CMAKE_INSTALL_PREFIX}/include
         FILES_MATCHING PATTERN "*.hpp"
                        PATTERN "*.h"
)
INSTALL (DIRECTORY
         "${PROJECT_SOURCE_DIR}/src/resilience/impl"
         DESTINATION ${CMAKE_INSTALL_PREFIX}/include
         FILES_MATCHING PATTERN "*.hpp"
                        PATTERN "*.h"
)
INSTALL (DIRECTORY
         "${PROJECT_SOURCE_DIR}/src/resilience/veloc"
         DESTINATION ${CMAKE_INSTALL_PREFIX}/include
         FILES_MATCHING PATTERN "*.hpp"
                        PATTERN "*.h"
)
INSTALL (DIRECTORY
         "${PROJECT_SOURCE_DIR}/src/resilience/util"
         DESTINATION ${CMAKE_INSTALL_PREFIX}/include
         FILES_MATCHING PATTERN "*.hpp"
                        PATTERN "*.h"
)

if (USE_HDF5)
    target_sources(resilience PRIVATE
              ${PROJECT_SOURCE_DIR}/src/resilience/impl/Kokkos_HDF5Space.cpp)
endif()

if (Kokkos_ENABLE_Cuda)
    target_sources(resilience PRIVATE
              ${PROJECT_SOURCE_DIR}/src/resilience/Cuda/Kokkos_ResCuda_Impl.cpp
              ${PROJECT_SOURCE_DIR}/src/resilience/Cuda/Kokkos_ResCudaSpace.cpp)
    INSTALL (DIRECTORY
         "${PROJECT_SOURCE_DIR}/src/resilience/Cuda"
         DESTINATION ${CMAKE_INSTALL_PREFIX}/include
         FILES_MATCHING PATTERN "*.hpp"
                        PATTERN "*.h"
    )
endif()

