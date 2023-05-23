#ifndef RES_CUDA_SUBSCRIBER_CPP
#define RES_CUDA_SUBSCRIBER_CPP

#include <Kokkos_Macros.hpp>
#if defined(KR_ENABLE_CUDA)

#include "ResCudaSubscriber.hpp"

namespace KokkosResilience {

int CudaResilientSubscriber::resilient_duplicate_counter = 0;

std::unordered_map< CudaResilientSubscriber::key_type, CombineDuplicatesBase * > CudaResilientSubscriber::duplicates_map;
std::unordered_map< CudaResilientSubscriber::key_type, std::unique_ptr< CombineDuplicatesBase > > CudaResilientSubscriber::duplicates_cache;

}

#endif //defined(KR_ENABLE_CUDA)
#endif //RES_CUDA_SUBSCRIBER_CPP
