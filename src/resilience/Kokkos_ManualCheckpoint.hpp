#ifndef __KOKKOS_RESILIENCE__
#define __KOKKOS_RESILIENCE__

#ifdef KOKKOS_ENABLE_MANUAL_CHECKPOINT
   #include <impl/Kokkos_StdFileSpace.hpp>

   #ifdef KOKKOS_ENABLE_HDF5 
      #include <impl/Kokkos_HDF5Space.hpp>
   #endif
#endif


#ifdef KOKKOS_ENABLE_RES_CUDA
#include <Kokkos_ResCuda.hpp>
#include <Kokkos_Cuda_ResParallel.hpp>
#endif

#endif

