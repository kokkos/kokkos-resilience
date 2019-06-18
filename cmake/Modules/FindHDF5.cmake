find_package(MPI)

find_path(_hdf5_root
          NAMES include/hdf5.h
          HINTS $ENV{SEMS_HDF5_ROOT} $ENV{HDF5_ROOT} ${HDF5_ROOT} ${HDF5_DIR}
          )

find_library(_hdf5_lib
             NAMES libhdf5.so
             HINTS ${_hdf5_root}/lib ${_hdf5_root}/lib64)

find_path(_hdf5_include_dir
          NAMES hdf5.h
          HINTS ${_hdf5_root}/include)

if ((NOT ${_hdf5_root})
        OR (NOT ${_hdf5_lib})
        OR (NOT ${_hdf5_include_dir}))
  set(_fail_msg "Could NOT find HDF5 (set HDF5_DIR or HDF5_ROOT to point to install)")
elseif ((NOT ${MPI_FOUND}) OR (NOT ${MPI_CXX_FOUND}))
  set(_fail_msg "Could NOT find HDF5 (missing MPI)")
else()
  set(_fail_msg "Could NOT find HDF5")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(HDF5 ${_fail_msg}
                                  _hdf5_root
                                  _hdf5_lib
                                  _hdf5_include_dir
                                  MPI_FOUND
                                  MPI_CXX_FOUND
                                  )

add_library(HDF5::HDF5 UNKNOWN IMPORTED)
set_target_properties(HDF5::HDF5 PROPERTIES
                      IMPORTED_LOCATION ${_hdf5_lib}
                      INTERFACE_INCLUDE_DIRECTORIES ${_hdf5_include_dir}
                      )

set(HDF5_DIR ${_hdf5_root})

mark_as_advanced(
  _hdf5_library
  _hdf5_include_dir
)
