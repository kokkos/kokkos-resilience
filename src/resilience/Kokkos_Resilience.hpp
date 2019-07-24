#ifndef INC_RESILIENCE_RESILIENCE_HPP
#define INC_RESILIENCE_RESILIENCE_HPP

#include <resilience/config/config.hpp>

#include "context.hpp"

#ifdef KOKKOS_ENABLE_MANUAL_CHECKPOINT
   #include "Kokkos_ManualCheckpoint.hpp"
#endif

#ifdef KOKKOS_ENABLE_AUTOMATIC_CHECKPOINT
   #include "Kokkos_AutomaticCheckpoint.hpp"
#endif

#ifdef KOKKOS_ENABLE_RES_CUDA
   #include "cuda/Kokkos_ResCuda.hpp"
   #include "cuda/Kokkos_ResCudaSpace.hpp"
   #include "cuda/Kokkos_Cuda_ResParallel.hpp"
#endif


#endif  // INC_RESILIENCE_RESILIENCE_HPP
