/*
 *
 *                        Kokkos v. 3.0
 *       Copyright (2020) National Technology & Engineering
 *               Solutions of Sandia, LLC (NTESS).
 *
 * Under the terms of Contract DE-NA0003525 with NTESS,
 * the U.S. Government retains certain rights in this software.
 *
 * Kokkos is licensed under 3-clause BSD terms of use:
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 * 1. Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the Corporation nor the names of the
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Questions? Contact Christian R. Trott (crtrott@sandia.gov)
 */

#ifndef INC_RESILIENCE_OPENMP_DUPLICATE_MAP_TRAVERSALS_HPP
#define INC_RESILIENCE_OPENMP_DUPLICATE_MAP_TRAVERSALS_HPP

#include <Kokkos_Macros.hpp>
#include "resilience/ErrorHandler.hpp"
#if defined(KOKKOS_ENABLE_OPENMP)

#include <omp.h>
#include <iostream>

#include <Kokkos_Core.hpp>

#include "Resilient_OpenMP_Subscriber.hpp"
#include "Resilient_OpenMP_Error_Injector.hpp"

namespace KokkosResilience{
	
  inline bool combine_resilient_duplicates() {
    bool success = true;
    // Combines all duplicates
    // Go over the Subscriber map, execute all the CombinerBase elements
    // If any duplicate fails to find a match, breaks
    for (auto&& combiner : KokkosResilience::ResilientDuplicatesSubscriber::duplicates_map) {
      success = combiner.second->execute();
      if(!success) break;
    }
    return success;
  }
 
#if defined KR_DETERMINISTIC_ERROR_INJECTION || defined KR_GEOMETRIC_ERROR_INJECTION  
  inline void inject_error_duplicates() {

    if (global_error_settings){
      //Per kernel, seed the first inject index     
      KokkosResilience::ErrorInjectionTracking::global_next_inject 
	      = KokkosResilience::global_error_settings->geometric(KokkosResilience::ErrorInjectionTracking::random_gen);
      // Inject error into duplicates, deterministically or randomly chosen at compile time
      // Go over the Subscriber map, inject into all the CombinerBase elements
      for (auto&& combiner : KokkosResilience::ResilientDuplicatesSubscriber::duplicates_map) {
        combiner.second->inject_error();
      }
    }
  }
#endif

} // namespace KokkosResilience

#endif // KOKKOS_ENABLE_OPENMP
#endif // INC_RESILIENCE_OPENMP_OPENMP_DUPLICATE_MAP_TRAVERSALS_HPP
