#ifndef INC_RESILIENCE_RESILIENCE_HPP
#define INC_RESILIENCE_RESILIENCE_HPP

#include <resilience/config/Config.hpp>

#include "Context.hpp"

#ifdef KOKKOS_ENABLE_MANUAL_CHECKPOINT
   #include "ManualCheckpoint.hpp"
#endif

#ifdef KOKKOS_ENABLE_AUTOMATIC_CHECKPOINT
   #include "AutomaticCheckpoint.hpp"
#endif

#ifdef KR_ENABLE_CUDA
   #include "cuda/ResCuda.hpp"
   #include "cuda/ResCudaSpace.hpp"
   #include "cuda/CudaResParallel.hpp"
#endif


#endif  // INC_RESILIENCE_RESILIENCE_HPP
