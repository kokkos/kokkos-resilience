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

#ifndef INC_RESILIENCE_OPENMP_OPENMPRESPARALLEL_HPP
#define INC_RESILIENCE_OPENMP_OPENMPRESPARALLEL_HPP

#include <Kokkos_Macros.hpp>
#include "resilience/ErrorHandler.hpp"
#if defined(KOKKOS_ENABLE_OPENMP)

#include <omp.h>
#include <iostream>

#include <Kokkos_Core.hpp>

#include "Resilient_OpenMP_Subscriber.hpp"

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
/************************* ERROR INSERTION TEST CODE ************************/
/*--------------------------------------------------------------------------*/

namespace KokkosResilience{

  inline void inject_error_duplicates() {

    if (global_error_settings){
      //Per kernel, seed the first inject index	    
      KokkosResilience::ErrorInject::global_next_inject = KokkosResilience::global_error_settings->geometric(KokkosResilience::ErrorInject::random_gen);	      ;

      // Inject geometrically distributed error into all duplicates
      // Go over the Subscriber map, inject into all the CombinerBase elements
      for (auto&& combiner : KokkosResilience::ResilientDuplicatesSubscriber::duplicates_map) {
        combiner.second->inject_error();
      }
    }
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
  using Policy       = Kokkos::RangePolicy<Traits...>;
  using WorkTag      = typename Policy::work_tag;
  using LaunchBounds = typename Policy::launch_bounds;
  using Member       = typename Policy::member_type;

  const FunctorType &  m_functor;
  const Policy m_policy;

  ParallelFor() = delete ;
  ParallelFor & operator = ( const ParallelFor & ) = delete ;

  using surrogate_policy = Kokkos::RangePolicy < Kokkos::OpenMP, WorkTag, LaunchBounds>;

#ifdef KR_ENABLE_WRAPPER
  auto make_wrapper (int64_t work_size, int64_t offset, const FunctorType &functor_copy_0, const FunctorType &functor_copy_1) const{
    if constexpr (std::is_void_v<WorkTag>){
      auto wrapper_functor = [&, work_size, offset](int64_t i){
        if (i < work_size)
        {
          m_functor (i + offset);
        }
        else if (( work_size <= i) && (i < (2 * work_size)))
        {
          functor_copy_0 (i + offset - work_size);
        }
        else
        {
          functor_copy_1 (i + offset - ( 2 * work_size));
        }
      };
      return wrapper_functor;
    }else if constexpr (!std::is_void_v<WorkTag>)
    { 
      auto wrapper_functor = [&, work_size, offset](WorkTag work_tag, int64_t i){
        if (i < work_size)
        {
          m_functor (work_tag, i + offset);
        }
        else if (( work_size <= i) && (i < (2 * work_size)))
        {
          functor_copy_0 (work_tag, i + offset - work_size);
        }
        else
        {
          functor_copy_1 (work_tag, i + offset - ( 2 * work_size));
        }
      };
      return wrapper_functor;
    }
  }
#endif

 public:
  inline void execute() const {
    //! The execution() function in this class performs an OpenMP execution of parallel for with
    //! modular redundancy. Non-constant views equipped with the triggering subscribers are
    //! duplicated and three concurrent executions divided equally between the available pool
    //! of OpenMP threads proceed. Duplicate views are combined back into a single view by calling
    //! a combiner to majority vote on the correct values. This process is optionally repeated until
    //! a value is voted correct or a given number of attempts is exceeded.
    //! There are some subtleties regarding which views are copied per kernel in the default subscriber
    //! See KokkosResilience::ResilienctDuplicatesSubscriber::duplicates_cache for details

    bool success = 0; //! This bool indicates that all views successfully reached a consensus.

      surrogate_policy wrapper_policy;
#ifdef KR_ENABLE_TMR
      wrapper_policy = surrogate_policy(m_policy.begin(), m_policy.end());
      // Trigger Subscriber constructors
      KokkosResilience::ResilientDuplicatesSubscriber::in_resilient_parallel_loop = true;
      auto functor_copy_0 = m_functor;
      auto functor_copy_1 = m_functor;
      KokkosResilience::ResilientDuplicatesSubscriber::in_resilient_parallel_loop = false;
#endif

#ifdef KR_ENABLE_WRAPPER
      auto work_size = m_policy.end() - m_policy.begin();
      auto offset = m_policy.begin();
      wrapper_policy = surrogate_policy(0, 3 * work_size );

      // Trigger Subscriber constructors
      KokkosResilience::ResilientDuplicatesSubscriber::in_resilient_parallel_loop = true;
      auto functor_copy_0 = m_functor;
      auto functor_copy_1 = m_functor;
      KokkosResilience::ResilientDuplicatesSubscriber::in_resilient_parallel_loop = false;

      auto wrapper_functor = make_wrapper (work_size, offset, functor_copy_0, functor_copy_1);
      
#endif

#ifdef KR_ENABLE_TMR

      // TMR execution with no wrapper scheduling

      Impl::ParallelFor< decltype(m_functor) , surrogate_policy, Kokkos::OpenMP > closure0( m_functor , wrapper_policy );
      Impl::ParallelFor< decltype(m_functor) , surrogate_policy, Kokkos::OpenMP > closure1( functor_copy_0 , wrapper_policy );
      Impl::ParallelFor< decltype(m_functor) , surrogate_policy, Kokkos::OpenMP > closure2( functor_copy_1 , wrapper_policy );

      closure0.execute();
      closure1.execute();
      closure2.execute();


      const auto start{std::chrono::steady_clock::now()};
      KokkosResilience::inject_error_duplicates();
      const auto stop{std::chrono::steady_clock::now()};
      KokkosResilience::ErrorTimerSettings::elapsed_seconds = KokkosResilience::ErrorTimerSettings::elapsed_seconds + (std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start));
      KokkosResilience::ErrorTimerSettings::total_error_time = KokkosResilience::ErrorTimerSettings::total_error_time + KokkosResilience::ErrorTimerSettings::elapsed_seconds;

      // Combine the duplicate views and majority vote on correctness
      success = KokkosResilience::combine_resilient_duplicates();
      // Does not clear the cache map, user must clear cache map before Kokkos::finalize()
      KokkosResilience::clear_duplicates_map();

#endif

#ifdef KR_ENABLE_DMR

      //DMR with failover to TMR on error
      wrapper_policy = surrogate_policy(m_policy.begin(), m_policy.end());

      // Trigger Subscriber constructors
      KokkosResilience::ResilientDuplicatesSubscriber::in_resilient_parallel_loop = true;
      auto functor_copy_0 = m_functor;
      KokkosResilience::ResilientDuplicatesSubscriber::in_resilient_parallel_loop = false;

      // DMR execution with no wrapper scheduling, on failover TMR
      Impl::ParallelFor< decltype(m_functor) , surrogate_policy, Kokkos::OpenMP > closure0(m_functor , wrapper_policy );
      Impl::ParallelFor< decltype(m_functor) , surrogate_policy, Kokkos::OpenMP > closure1(functor_copy_0 , wrapper_policy );

      closure0.execute();
      closure1.execute();

      const auto start{std::chrono::steady_clock::now()};
      KokkosResilience::inject_error_duplicates();
      const auto stop{std::chrono::steady_clock::now()};
      KokkosResilience::ErrorTimerSettings::elapsed_seconds = KokkosResilience::ErrorTimerSettings::elapsed_seconds + (std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start));
      KokkosResilience::ErrorTimerSettings::total_error_time = KokkosResilience::ErrorTimerSettings::total_error_time + KokkosResilience::ErrorTimerSettings::elapsed_seconds;
      
      // Combine the duplicate views, majority vote not triggered due to CMAKE macro
      success = KokkosResilience::combine_resilient_duplicates();

      if (!success)
      {
        KokkosResilience::ResilientDuplicatesSubscriber::dmr_failover_to_tmr = true;
        KokkosResilience::ResilientDuplicatesSubscriber::in_resilient_parallel_loop = true;
	auto functor_copy_1 = m_functor;
        KokkosResilience::ResilientDuplicatesSubscriber::in_resilient_parallel_loop = false;

        Impl::ParallelFor< decltype(m_functor) , surrogate_policy, Kokkos::OpenMP > closure2(functor_copy_1 , wrapper_policy );

        start=std::chrono::steady_clock::now();
        KokkosResilience::inject_error_duplicates();
        stop=std::chrono::steady_clock::now();
        KokkosResilience::ErrorTimerSettings::elapsed_seconds = KokkosResilience::ErrorTimerSettings::elapsed_seconds + (std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start));
        KokkosResilience::ErrorTimerSettings::total_error_time = KokkosResilience::ErrorTimerSettings::total_error_time + KokkosResilience::ErrorTimerSettings::elapsed_seconds;

        success = KokkosResilience::combine_resilient_duplicates();
        KokkosResilience::clear_duplicates_map();
        KokkosResilience::ResilientDuplicatesSubscriber::dmr_failover_to_tmr = false;
      }
      KokkosResilience::clear_duplicates_map();
#endif
#ifdef KR_ENABLE_WRAPPER

      // TMR with kernel fusion
      // Functor is fused, with iterations bound to duplicated functors in 3x range
      Impl::ParallelFor< decltype(wrapper_functor) , surrogate_policy, Kokkos::OpenMP > closure( wrapper_functor , wrapper_policy );

      closure.execute();

      const auto start{std::chrono::steady_clock::now()};
      KokkosResilience::inject_error_duplicates();
      const auto stop{std::chrono::steady_clock::now()};
      KokkosResilience::ErrorTimerSettings::elapsed_seconds = KokkosResilience::ErrorTimerSettings::elapsed_seconds + (std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start));
      KokkosResilience::ErrorTimerSettings::total_error_time = KokkosResilience::ErrorTimerSettings::total_error_time + KokkosResilience::ErrorTimerSettings::elapsed_seconds;

      // Combine the duplicate views and majority vote on correctness
      success = KokkosResilience::combine_resilient_duplicates();
      KokkosResilience::clear_duplicates_map();
#endif

    if(success==0){
      // Abort if no agreement in duplicates
      auto &handler = KokkosResilience::get_unrecoverable_data_corruption_handler();
      handler(0);
    }

  } // execute

  inline ParallelFor(const FunctorType & arg_functor,
                     Policy arg_policy)
                    : m_functor(arg_functor),
                      m_policy(arg_policy) {
  }

};// range policy implementation

} // namespace Impl
} // namespace Kokkos

#endif // KOKKOS_ENABLE_OPENMP
#endif // INC_RESILIENCE_OPENMP_OPENMPRESPARALLEL_HPP
