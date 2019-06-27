#ifndef __KOKKOS_RESILIENCE__
#define __KOKKOS_RESILIENCE__

#ifdef KOKKOS_ENABLE_MANUAL_CHECKPOINT
   #include <impl/Kokkos_StdFileSpace.hpp>

   #ifdef KOKKOS_ENABLE_HDF5 
      #include <impl/Kokkos_HDF5Space.hpp>
   #endif
#endif


#endif

