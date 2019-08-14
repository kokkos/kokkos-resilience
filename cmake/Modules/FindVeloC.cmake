find_package(MPI)

find_path(_veloc_root
          NAMES include/veloc.h
          HINTS $ENV{VeloC_ROOT} ${VeloC_ROOT} ${VeloC_DIR}
          )

find_library(_veloc_client_lib
             NAMES libveloc.a veloc libveloc-client.so
             HINTS ${_veloc_root}/lib ${_veloc_root}/lib64)

find_library(_veloc_module_lib
             NAMES libveloc.a veloc libveloc-modules.so
             HINTS ${_veloc_root}/lib ${_veloc_root}/lib64)

find_library(_veloc_axl
             NAMES libaxl.a axl
             HINTS ${_veloc_root}/lib ${_veloc_root}/lib64)

find_library(_veloc_er
             NAMES liber.a er
             HINTS ${_veloc_root}/lib ${_veloc_root}/lib64)

find_library(_veloc_rankstr
             NAMES librankstr.a rankstr
             HINTS ${_veloc_root}/lib ${_veloc_root}/lib64)

find_library(_veloc_kvtree
             NAMES libkvtree.a kvtree
             HINTS ${_veloc_root}/lib ${_veloc_root}/lib64)

find_path(_veloc_include_dir
          NAMES veloc.h
          HINTS ${_veloc_root}/include)

if ((NOT ${_veloc_root})
     OR (NOT ${_veloc_client_lib})
     OR (NOT ${_veloc_module_lib})
     OR (NOT ${_veloc_include_dir}))
  set(_fail_msg "Could NOT find VeloC (set VeloC_DIR or VeloC_ROOT to point to install)")
else()
  set(_fail_msg "Could NOT find VeloC")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(VeloC ${_fail_msg}
                                  _veloc_root
                                  _veloc_client_lib
                                  _veloc_module_lib
                                  _veloc_include_dir
                                  MPI_FOUND
                                  MPI_CXX_FOUND
                                  )

add_library(veloc::axl UNKNOWN IMPORTED)
set_target_properties(veloc::axl PROPERTIES
                      IMPORTED_LOCATION ${_veloc_axl}
                      )

add_library(veloc::er UNKNOWN IMPORTED)
set_target_properties(veloc::er PROPERTIES
                      IMPORTED_LOCATION ${_veloc_er}
                      )

add_library(veloc::rankstr UNKNOWN IMPORTED)
set_target_properties(veloc::rankstr PROPERTIES
                      IMPORTED_LOCATION ${_veloc_rankstr}
                      )

add_library(veloc::kvtree UNKNOWN IMPORTED)
set_target_properties(veloc::kvtree PROPERTIES
                      IMPORTED_LOCATION ${_veloc_rankstr}
                      )

add_library(veloc::veloc_modules UNKNOWN IMPORTED)
set_target_properties(veloc::veloc_modules PROPERTIES
                      IMPORTED_LOCATION ${_veloc_module_lib}
                      )

add_library(veloc::veloc UNKNOWN IMPORTED)
set_target_properties(veloc::veloc PROPERTIES
                      IMPORTED_LOCATION ${_veloc_client_lib}
                      INTERFACE_INCLUDE_DIRECTORIES ${_veloc_include_dir}
                      INTERFACE_LINK_LIBRARIES "veloc::veloc_modules;MPI::MPI_CXX"
                      )

if (NOT VELOC_BAREBONE)
  set_target_properties(veloc::veloc PROPERTIES
                        INTERFACE_LINK_LIBRARIES "veloc::axl;veloc::er;veloc::rankstr"
                        )
endif()

set(VeloC_DIR ${_veloc_root})

mark_as_advanced(
  _veloc_library
  _veloc_include_dir
)
