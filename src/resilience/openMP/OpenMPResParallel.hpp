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
  using WorkRange = typename Policy::WorkRange;
  using Member    = typename Policy::member_type;

  OpenMPExec* m_instance;
  const FunctorType &  m_functor;
  const Policy m_policy;

  typedef Kokkos::RangePolicy<Kokkos::OpenMP, WorkTag, WorkRange> surrogate_policy;

  template <class TagType>
  inline static
      typename std::enable_if<std::is_same<TagType, void>::value>::type
      exec_range(const FunctorType& functor, const Member ibeg,
                 const Member iend) {

#ifdef KOKKOS_ENABLE_AGGRESSIVE_VECTORIZATION
#ifdef KOKKOS_ENABLE_PRAGMA_IVDEP
#pragma ivdep
#endif
#endif
    for (Member iwork = ibeg; iwork < iend; ++iwork) {
      functor(iwork);
    }
  }

  template <class TagType>
  inline static
      typename std::enable_if<!std::is_same<TagType, void>::value>::type
      exec_range(const FunctorType& functor, const Member ibeg,
                 const Member iend) {
    const TagType t{};
#ifdef KOKKOS_ENABLE_AGGRESSIVE_VECTORIZATION
#ifdef KOKKOS_ENABLE_PRAGMA_IVDEP
#pragma ivdep
#endif
#endif
    for (Member iwork = ibeg; iwork < iend; ++iwork) {
      functor(t, iwork);
    }
  }

  inline void
  exec_work (const FunctorType& functor,
             const surrogate_policy & lPolicy,
             bool dynamic_value,
             int exec_num_threads) const {

    //Initial setup, returns for each thread
    HostThreadTeamData& data = *(m_instance->get_thread_data());

    data.set_work_partition(lPolicy.end() - lPolicy.begin(),
                                     lPolicy.chunk_size());

    if (OpenMP::impl_thread_pool_size() < 3) {

      if (dynamic_value) {
        // Make sure work partition is set before stealing
        if (data.pool_rendezvous()) data.pool_rendezvous_release();
      }

      std::pair<int64_t, int64_t> range(0, 0);

      do {
        range = dynamic_value ? data.get_work_stealing_chunk()
                             : data.get_work_partition();

        ParallelFor::template exec_range<WorkTag>(
            functor, range.first + lPolicy.begin(),
            range.second + lPolicy.begin());

      } while (dynamic_value && 0 <= range.first);

    }
    else {
      // Per-thread scheduler modifications for triplicate execution
      int64_t chunk_min = ((lPolicy.end() - lPolicy.begin()) +
                           std::numeric_limits<int>::max()) /
                          std::numeric_limits<int>::max();

      int64_t work_end = lPolicy.end() - lPolicy.begin();

      int64_t chunk = lPolicy.chunk_size();

      // No increase to chunk size
      int64_t work_chunk = std::max(chunk, chunk_min);

      // League size may change
      int league_size = OpenMP::impl_thread_pool_size();

      // num is number of work chunks - same for all 3 executions
      int num = (work_end + work_chunk - 1) / work_chunk;

      // part based on which execution functor thread possesses - different per
      // execution
      int part = (num + exec_num_threads - 1) / exec_num_threads;

      // This doesn't happen unless the schedule is set to dynamic
      // This case is currently unavailable
      if (dynamic_value) {
        // Make sure work partition is set before stealing
        if (data.pool_rendezvous()) data.pool_rendezvous_release();
      }

      std::pair<int64_t, int64_t> range(0, 0);

      // Resetting the first range to rescheduled first range
      if (omp_get_thread_num() < league_size / 3) {
        range.first = part * omp_get_thread_num();
      }

      if (omp_get_thread_num() >= league_size / 3 &&
          omp_get_thread_num() < (2 * league_size) / 3) {
        range.first = part * (omp_get_thread_num() - (league_size / 3));
      }

      if (omp_get_thread_num() < league_size &&
          omp_get_thread_num() >= (2 * league_size) / 3) {
        range.first = part * (omp_get_thread_num() - ((2 * league_size) / 3));
      }

      range.second = range.first + part;
      range.second = range.second < work_end ? range.second : work_end;

      do {
        // Dynamic case not available
        // range = dynamic_value ? data.get_work_stealing_chunk()
        //                   : data.get_work_partition();

        range.first *= work_chunk;
        range.second *= work_chunk;
        range.second = range.second < work_end ? range.second : work_end;

        ParallelFor::template exec_range<WorkTag>(
            functor, range.first + lPolicy.begin(),
            range.second + lPolicy.begin());

      } while (dynamic_value && 0 <= range.first);
    }
  }

 public:
  inline void execute() const {

    int repeats = 5;
    bool success = 0;

    while(success==0 && repeats > 0){

      static constexpr bool is_dynamic = std::is_same<typename Policy::schedule_type::type,
                                Kokkos::Dynamic>::value;

      surrogate_policy lPolicy[3];
      for (int i = 0; i < 3; i++) {
        lPolicy[i] = surrogate_policy(m_policy.begin(), m_policy.end());
      }

      // parallel_for turns off shared allocation tracking, toggle it back on for ViewHooks
      Kokkos::Impl::shared_allocation_tracking_enable();

      // Trigger Subscriber constructors
      KokkosResilience::ResilientDuplicatesSubscriber::in_resilient_parallel_loop = true;
      auto m_functor_0 = m_functor;
      auto m_functor_1 = m_functor;
      auto m_functor_2 = m_functor;
      KokkosResilience::ResilientDuplicatesSubscriber::in_resilient_parallel_loop = false;

      // toggle the shared allocation tracking off again
      // Allows for user-intended view behavior in main body of parallel_for
      Kokkos::Impl::shared_allocation_tracking_disable();

      if (OpenMP::in_parallel()) {
        exec_range<WorkTag>(m_functor, m_policy.begin(), m_policy.end());
      } else {
        OpenMPExec::verify_is_master("Kokkos::OpenMP parallel_for");

        // Three consecutive executions if less than three threads total
        if (OpenMP::impl_thread_pool_size() < 3) {

          int exec_num_threads = OpenMP::impl_thread_pool_size();
          exec_work(m_functor_0, lPolicy[0], is_dynamic, exec_num_threads);
          exec_work(m_functor_1, lPolicy[1], is_dynamic, exec_num_threads);
          exec_work(m_functor_2, lPolicy[2], is_dynamic, exec_num_threads);

        } else {
          //Three concurrent executions with 1/3 threads each if >3 threads
#pragma omp parallel num_threads(OpenMP::impl_thread_pool_size())
          {
            if (omp_get_thread_num() < OpenMP::impl_thread_pool_size() / 3) {
              int exec_num_threads = OpenMP::impl_thread_pool_size() / 3;
              exec_work(m_functor_0, lPolicy[0], is_dynamic, exec_num_threads);
            }

            if (omp_get_thread_num() >= OpenMP::impl_thread_pool_size() / 3 &&
                omp_get_thread_num() <
                    (2 * OpenMP::impl_thread_pool_size()) / 3) {
              int exec_num_threads = ((2 * OpenMP::impl_thread_pool_size())/3) -
                                     (OpenMP::impl_thread_pool_size()/3);
              exec_work(m_functor_1, lPolicy[1], is_dynamic, exec_num_threads);
            }

            if (omp_get_thread_num() < OpenMP::impl_thread_pool_size() &&
                omp_get_thread_num() >=
                    (2 * OpenMP::impl_thread_pool_size()) / 3) {
              int exec_num_threads = OpenMP::impl_thread_pool_size() -
                                     ((2 * OpenMP::impl_thread_pool_size())/3);
              exec_work(m_functor_2, lPolicy[2], is_dynamic, exec_num_threads);
            }

          }  // pragma omp
        }    //else
      } // omp-parallel else
      Kokkos::fence();
      //KokkosResilience::print_duplicates_map();
      //Kokkos::fence();

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

  inline ParallelFor(const FunctorType& arg_functor, Policy arg_policy)
      : m_instance(t_openmp_instance),
        m_functor(arg_functor),
        m_policy(arg_policy) {
  }

};// range policy implementation

} // namespace Impl
} // namespace Kokkos

#endif // KOKKOS_ENABLE_OPENMP
#endif // INC_RESILIENCE_OPENMP_OPENMPRESPARALLEL_HPP