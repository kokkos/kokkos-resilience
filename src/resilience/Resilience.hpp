#ifndef INC_RESILIENCE_RESILIENCE_HPP
#define INC_RESILIENCE_RESILIENCE_HPP

#include <resilience/config/Config.hpp>

#include "Context.hpp"
#include "ManualCheckpoint.hpp"

#ifdef KR_ENABLE_VELOC
#include "veloc/VelocBackend.hpp"
#include "AutomaticCheckpoint.hpp"
#endif

#ifdef KR_ENABLE_CUDA
   #include "cuda/ResCuda.hpp"
   #include "cuda/ResCudaSpace.hpp"
   #include "cuda/CudaResParallel.hpp"
#endif


#endif  // INC_RESILIENCE_RESILIENCE_HPP
