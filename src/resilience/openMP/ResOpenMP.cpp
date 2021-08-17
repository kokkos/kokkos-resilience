/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#include <Kokkos_Macros.hpp>
#if defined(KOKKOS_ENABLE_OPENMP)

#include <cstdio>
#include <cstdlib>

#include <limits>
#include <iostream>
#include <vector>

#include <Kokkos_Core.hpp>

#include <impl/Kokkos_Error.hpp>
#include <impl/Kokkos_CPUDiscovery.hpp>
#include <impl/Kokkos_Profiling_Interface.hpp>

#include <impl/Kokkos_Traits.hpp>
#include <Kokkos_UniqueToken.hpp>
#include <impl/Kokkos_ConcurrentBitset.hpp>

#include <Kokkos_OpenMP.hpp>
#include <OpenMP/Kokkos_OpenMP_Exec.hpp>
#include "ResOpenMP.hpp"

/*------------------------------------------------------------------------*/

namespace KokkosResilience {

int ResOpenMP::concurrency()
{ return OpenMP::concurrency();}

// Has been initialized: possible noexcept syntax in Kokkos::OpenMP
inline bool ResOpenMP::impl_is_initialized() noexcept {
  return Kokkos::OpenMP::impl_is_initialized();
}

void ResOpenMP::impl_finalize() {
  Kokkos::OpenMP::impl_finalize();

  #if defined(KOKKOS_ENABLE_PROFILING)
Kokkos::Profiling::finalize();
  #endif
}

void ResOpenMP::impl_initialize(int thread_count) {
  Kokkos::OpenMP::impl_initialize( thread_count );

  #if defined(KOKKOS_ENABLE_PROFILING)
    Kokkos::Profiling::initialize();
  #endif

}

ResOpenMP::ResOpenMP()
  : OpenMP() {
}

void ResOpenMP::print_configuration( std::ostream & s , const bool )
{
  s << "KokkosResilience::ResOpenMP";

  const bool is_initialized = Kokkos::Impl::t_openmp_instance != nullptr;

  if (is_initialized) {
    Kokkos::Impl::OpenMPExec::verify_is_master("ResOpenMP::print_configuration");

    const int numa_count      = 1;
    const int core_per_numa   = Kokkos::Impl::g_openmp_hardware_max_threads;
    const int thread_per_core = 1;

    s << " thread_pool_topology[ " << numa_count << " x " << core_per_numa
      << " x " << thread_per_core << " ]" << std::endl;
  } else {
    s << " not initialized" << std::endl;
  }
}

const char* ResOpenMP::name() { return "ResOpenMP"; }

} // namespace KokkosResilience

#else

void KOKKOS_CORE_SRC_OPENMP_EXEC_PREVENT_LINK_ERROR() {}

/*------------------------------------------------------------------------*/

#endif // KOKKOS_ENABLE_OPENMP

