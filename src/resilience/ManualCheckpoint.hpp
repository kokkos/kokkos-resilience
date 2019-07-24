#ifndef __KOKKOS_RESILIENCE__
#define __KOKKOS_RESILIENCE__

#ifdef KOKKOS_ENABLE_MANUAL_CHECKPOINT
   #include "resilience/stdio/StdFileSpace.hpp"

   #ifdef KOKKOS_ENABLE_HDF5
      #include "hdf5/HDF5Space.hpp"
   #endif
#endif


#endif

