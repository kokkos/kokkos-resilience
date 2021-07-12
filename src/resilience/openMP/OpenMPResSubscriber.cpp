#ifndef INC_RESILIENCE_OPENMP_OPENMPRESSUBSCRIBER_CPP
#define INC_RESILIENCE_OPENMP_OPENMPRESSUBSCRIBER_CPP

#include <Kokkos_Macros.hpp>
#if defined(KOKKOS_ENABLE_OPENMP) //&& defined (KR_ENABLE_ACTIVE_EXECUTION_SPACE)

#include <omp.h>
#include <iostream>
#include <OpenMP/Kokkos_OpenMP_Exec.hpp>
#include <impl/Kokkos_FunctorAdapter.hpp>

#include <KokkosExp_MDRangePolicy.hpp>

#include <Kokkos_Parallel.hpp>
#include <OpenMP/Kokkos_OpenMP_Parallel.hpp>
#include <Kokkos_Core_fwd.hpp>
#include <cmath>
#include <map>
#include <typeinfo>
#include <unordered_map>
#include "OpenMPResSubscriber.hpp"

namespace KokkosResilience {

struct ResilientDuplicatesSubscriber;

// Try to initialize bool here, but did not work. Ugh.
bool ResilientDuplicatesSubscriber::in_resilient_parallel_loop = false;

//std::unordered_map < ResilientDuplicatesSubscriber::key_type, std::unique_ptr<CombineDuplicatesBase> > duplicates_map;

}

#endif //defined(KOKKOS_ENABLE_OPENMP) //&& defined (KR_ENABLE_ACTIVE_EXECUTION_SPACE)
#endif //INC_RESILIENCE_OPENMP_OPENMPRESSUBSCRIBER_CPP