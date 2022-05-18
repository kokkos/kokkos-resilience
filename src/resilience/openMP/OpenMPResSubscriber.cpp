#ifndef INC_RESILIENCE_OPENMP_OPENMPRESSUBSCRIBER_CPP
#define INC_RESILIENCE_OPENMP_OPENMPRESSUBSCRIBER_CPP

#include <Kokkos_Macros.hpp>
#if defined(KOKKOS_ENABLE_OPENMP)

#include "OpenMPResSubscriber.hpp"

namespace KokkosResilience {

bool ResilientDuplicatesSubscriber::in_resilient_parallel_loop = false;
bool ResilientDuplicatesSubscriber::dmr_failover_to_tmr = false;

std::unordered_map< ResilientDuplicatesSubscriber::key_type, CombineDuplicatesBase * > ResilientDuplicatesSubscriber::duplicates_map;
std::unordered_map< ResilientDuplicatesSubscriber::key_type, std::unique_ptr< CombineDuplicatesBase > > ResilientDuplicatesSubscriber::duplicates_cache;

}

#endif //defined(KOKKOS_ENABLE_OPENMP)
#endif //INC_RESILIENCE_OPENMP_OPENMPRESSUBSCRIBER_CPP