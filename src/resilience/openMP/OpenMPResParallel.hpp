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

#ifndef INC_RESILIENCE_OPENMP_OPENMPRESPARALLEL_HPP
#define INC_RESILIENCE_OPENMP_OPENMPRESPARALLEL_HPP

#include <Kokkos_Macros.hpp>
#if defined(KOKKOS_ENABLE_OPENMP)

#include <omp.h>
#include <iostream>

#include <OpenMP/Kokkos_OpenMP_Exec.hpp>
#include <impl/Kokkos_FunctorAdapter.hpp>

#include <KokkosExp_MDRangePolicy.hpp>

#include <Kokkos_Parallel.hpp>
#include <OpenMP/Kokkos_OpenMP_Parallel.hpp>

#include "OpenMPResSubscriber.hpp"

/*--------------------------------------------------------------------------*/
/************************** DUPLICATE COMBINER CALL *************************/
/*--------------------------------------------------------------------------*/

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

} // namespace KokkosResilience

/*--------------------------------------------------------------------------*/
/************************ RESILIENT PARALLEL FORS ***************************/
/*--------------------------------------------------------------------------*/

namespace Kokkos {
namespace Impl {

// Range policy implementation   
template <class FunctorType, class... Traits>
class ParallelFor< FunctorType
                 , Kokkos::RangePolicy< Traits... >
                 , KokkosResilience::ResOpenMP>{
 private:
  using Policy    = Kokkos::RangePolicy<Traits...>;
  using WorkTag   = typename Policy::work_tag;

  OpenMPExec* m_instance;
  const FunctorType &  m_functor;
  const Policy m_policy;

  ParallelFor() = delete ;
  ParallelFor & operator = ( const ParallelFor & ) = delete ;

  using surrogate_policy = Kokkos::RangePolicy < Kokkos::OpenMP, WorkTag >;

 public:
  inline void execute() const {

      // TODO: Massively change comment to reflect changed paradigm
      //! Somewhere (possibly after the } of execute) a long comment describe execute, such as:
      //! The execute() function in this class performs an OpenMP execution of parallel for
      //! with triple modular redundancy. Views equipped with the necessary subscribers are
      //! duplicated and three concurrent executions divided equally between the available pool
      //! of OpenMP threads proceed. Duplicate views are combined back into a single view by calling
      //! a combiner to majority vote on the correct values. This process is repeated until
      //! a value is voted correct or a given number of attempts is exceeded.

    int repeats = 5; //! This integer represents the maximum number of attempts to reach consensus allowed.
    bool success = 0; //! This bool indicates that all views successfully reached a consensus.

    while(success==0 && repeats > 0){


      // TODO: SHOULD THERE BE A GUARD ON THE END SIZE? AND IF SO WHAT BEHAVIOR DESIRED?
      surrogate_policy wrapper_policy;
      wrapper_policy = surrogate_policy(m_policy.begin(), 3 * m_policy.end());

      // Trigger Subscriber constructors
      KokkosResilience::ResilientDuplicatesSubscriber::in_resilient_parallel_loop = true;
      auto m_functor_0 = m_functor;
      auto m_functor_1 = m_functor;
      auto m_functor_2 = m_functor;
      KokkosResilience::ResilientDuplicatesSubscriber::in_resilient_parallel_loop = false;

      auto wrapper_functor = [&](auto i){
        if (i < m_policy.end())
        {
          m_functor_0 (i);
        }
        else if (( m_policy.end() <= i) && (i < 2 * m_policy.end()))
        {
          m_functor_1 (i - m_policy.end());
        }
        else
        {
           m_functor_2 (i - ( 2 * m_policy.end()));
        }
    // TODO: Massively change comment to reflect changed paradigm
    //! Somewhere (possibly after the } of execute) a long comment describe execute, such as:
    //! The execute() function in this class performs an OpenMP execution of parallel for
    //! with triple modular redundancy. Views equipped with the necessary subscribers are
    //! duplicated and three concurrent executions divided equally between the available pool
    //! of OpenMP threads proceed. Duplicate views are combined back into a single view by calling
    //! a combiner to majority vote on the correct values. This process is repeated until
    //! a value is voted correct or a given number of attempts is exceeded.
      };

      // toggle the shared allocation tracking off again
      // Allows for user-intended view behavior in main body of parallel_for
      //Kokkos::Impl::shared_allocation_tracking_disable();

      // TODO:
      // ALL THREAD SCHEDULING HANDLED BY KOKKOS HERE, ITERATION HANDLING BY US
      // Attempt to feed in a three-times as long range policy (wrapper-policy)
      // With a wrapped functor, so that the iterations are bound to the duplicated functors/views
      Impl::ParallelFor< decltype(wrapper_functor) , surrogate_policy, Kokkos::OpenMP > closure( wrapper_functor , wrapper_policy );

      // Execute it.
      closure.execute();



      Kokkos::fence();
      // KokkosResilience::print_duplicates_map();
      Kokkos::fence();

      // Combine the duplicate views and majority vote on correctness
      success = KokkosResilience::combine_resilient_duplicates();

      Kokkos::fence();

      KokkosResilience::clear_duplicates_map();
      repeats--;

    }// while (!success & repeats left)

    if(success==0 && repeats == 0){
    // Abort if 5 repeated tries at executing failed to find acceptable match
    Kokkos::abort("Aborted in parallel_for, resilience majority voting failed because each execution obtained a differing value.");
    }

  } // execute

  inline ParallelFor(const FunctorType & arg_functor,
                     Policy arg_policy)
      : m_instance(t_openmp_instance),
        m_functor(arg_functor),
        m_policy(arg_policy) {
  }

};// range policy implementation

} // namespace Impl
} // namespace Kokkos

#endif // KOKKOS_ENABLE_OPENMP
#endif // INC_RESILIENCE_OPENMP_OPENMPRESPARALLEL_HPP