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
#if defined(KOKKOS_ENABLE_OPENMP) //&& defined (KR_ENABLE_ACTIVE_EXECUTION_SPACE)

#include <omp.h>
#include <iostream>
#include <OpenMP/Kokkos_OpenMP_Exec.hpp>
#include <impl/Kokkos_FunctorAdapter.hpp>

#include <KokkosExp_MDRangePolicy.hpp>

#include <Kokkos_Parallel.hpp>
#include <OpenMP/Kokkos_OpenMP_Parallel.hpp>//main parallel

#include "OpenMPResSubscriber.hpp"

/*--------------------------------------------------------------------------*/
/********************  OVERALL COMBINER FUNCTION CALL  **********************/
/*--------------------------------------------------------------------------*/


namespace KokkosResilience{

inline bool combine_resilient_duplicates() {

  printf("Entered the combine resilient duplicates function. \n\n");
  fflush(stdout);

  bool success = true;
  // Combines all duplicates
  // Go over the Subscriber map, execute all the CombinerBase elements
  for (auto&& combiner : KokkosResilience::ResilientDuplicatesSubscriber::duplicates_map) {
    success = combiner.second->execute();
    if(!success) break;
  }

  //TODO: CHANGE TO GENTLE, FIX ERROR
  if (!success) {
    Kokkos::abort("Aborted in combine_resilient_duplicates");
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

    //printf("Thread %d in resilient Kokkos range-policy, executing work.\n", omp_get_thread_num());
    //fflush(stdout);

    //Initial setup, returns for each thread up to MAX_THREAD
    HostThreadTeamData& data = *(m_instance->get_thread_data());

    //Feeds in the total range and a chunk size, which is set in the work partition
    // to the max of the chunk size and
    //(length + std::numeric_limits<int>::max()) / std::numeric_limits<int>::max();
    //Also sets the steal rank =  m_team_base + m_team_alloc + m_team_size <= m_pool_size
    //                        ? m_team_base + m_team_alloc
    //                        : 0;
    // No access to these variables

    data.set_work_partition(lPolicy.end() - lPolicy.begin(),
                                     lPolicy.chunk_size());


    //printf("ThreadID: %d, Policy.end(): %d, Policy,begin(): %d, and Policy.chunk_size(): %d\n", omp_get_thread_num(), lPolicy.end(), lPolicy.begin(), lPolicy.chunk_size());
    //fflush(stdout);

    // HERE STARTS SCHEDULER MODIFICATIONS, CLEAN UP
    // to separate function (?), not triggered in case of < 3 threads.

    // Per-thread rescheduling.
    // TODO: TEST CORNER CASES BELOW
    // 1) Uneven thread split (3-3-2) CHECKED, FINE
    // 2) Range unevenly divided (e.g., the range-exceeding case in 3-3-2 with last 2 threads originally assigned 2 and 0 elements) CHECKED, FINE
    // 3) Massive size: Unchecked
    // 4) Range size < # threads available CHECKED, FINE

    int chunk_min = ((lPolicy.end()-lPolicy.begin()) + std::numeric_limits<int>::max()) /
                            std::numeric_limits<int>::max();

    int work_end  = lPolicy.end()-lPolicy.begin();
    //printf("ThreadID: %d, workend %d and chunkmin %d\n", omp_get_thread_num(), work_end, chunk_min);
    //fflush(stdout);

    int chunk = lPolicy.chunk_size();

    // Have not increased chunk size, chunk size is unit element work
    int work_chunk = std::max(chunk, chunk_min);

    // LEAGUE SIZE BASED ON DEBUGGER EVIDENCE
    int league_size = OpenMP::impl_thread_pool_size();

    // num is number of work chunks - same for all 3 executions
    int num  = (work_end + work_chunk - 1) / work_chunk;

    // Separate part based on which execution functor thread possesses
    // partitioning is different based on num_exec_threads
    int part = (num + exec_num_threads - 1) / exec_num_threads;

    // TODO: STEAL RANK? DYNAMIC ONLY
    // No procedure for invoking this in range policy implementation, unless dynamic triggered

    // This doesn't happen unless the schedule is set to dynamic, a rare occurrence
    //TODO: ACCOUNT FOR THIS, WILL BREAK CODE
    if (dynamic_value) {
      // Make sure work partition is set before stealing
      if (data.pool_rendezvous()) data.pool_rendezvous_release();
    }

    std::pair<int64_t, int64_t> range(0, 0);

    // Artificially resetting the first range to rescheduled first range
    // Always overridden? No, took out overrider, now it's necessary.
    if (omp_get_thread_num() < league_size / 3) {

      range.first = part * omp_get_thread_num();
    }

    if (omp_get_thread_num() >= league_size / 3 &&
        omp_get_thread_num() < (2 * league_size) / 3) {

      range.first = part * (omp_get_thread_num() - (league_size/3));
    }

    if (omp_get_thread_num() < league_size &&
        omp_get_thread_num() >= (2 * league_size) / 3) {

      range.first = part * (omp_get_thread_num() - ( ( 2 * league_size) / 3 ) );
    }

    range.second = range.first + part;
    range.second = range.second < work_end ? range.second : work_end;

    //printf("ThreadID %d in after setup, first range: %d - %d,\n", omp_get_thread_num(),range.first,range.second);
    //fflush(stdout);

    do {
      // If (dynamic_value), range=data.getworkstealing, else range=data.getworkpartition
      // Dynamic value is 0, this is triggering get work partition
      // Dynamic value is a rare occurrance specifically triggered by user.
      // TODO: ACCOUNT FOR THIS


      //range = dynamic_value ? data.get_work_stealing_chunk()
      //                   : data.get_work_partition();

      // This is overriding the scheduling from range above.
      // Once secure, find way to delete that scheduling, no need for 2x schedule

      range.first *= work_chunk;
      range.second *= work_chunk;
      range.second = range.second < work_end ? range.second : work_end;

      //printf("ThreadID %d in dowhile, updating work partition: %d - %d,\n", omp_get_thread_num(),range.first,range.second);
      //fflush(stdout);

      ParallelFor::template exec_range<WorkTag>(
                   functor, range.first + lPolicy.begin(),
                            range.second + lPolicy.begin());

    } while (dynamic_value && 0 <= range.first);
  }

 public:
  inline void execute() const {

    int repeats = 5;
    bool success = 0;

    while(success==0 && repeats > 0){
      printf("Thread %d in resilient Kokkos rp-for, before parallel pragma.\n", omp_get_thread_num());
      fflush(stdout);
      printf("This is execution number %d.\n",repeats);
      fflush(stdout);

      static constexpr bool is_dynamic = std::is_same<typename Policy::schedule_type::type,
                                Kokkos::Dynamic>::value;

      std::cout << "This is is_dynamic:" << is_dynamic << std::endl;

      surrogate_policy lPolicy[3];
      for (int i = 0; i < 3; i++) {
        lPolicy[i] = surrogate_policy(m_policy.begin(), m_policy.end());
      }



      // ViewHooks captures non-constant views and passes to duplicate_shared
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

//TODO: MAKE EXEC_WORK TRIGGER RANGE MITIGATION ONLY ON 3 OR MORE THREADS
        if (OpenMP::impl_thread_pool_size() < 3) {

          int exec_num_threads = OpenMP::impl_thread_pool_size();
          exec_work(m_functor_0, lPolicy[0], is_dynamic, exec_num_threads);
          exec_work(m_functor_1, lPolicy[1], is_dynamic, exec_num_threads);
          exec_work(m_functor_2, lPolicy[2], is_dynamic, exec_num_threads);

        } else {
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
    // TODO: Improve error message to give for label, change to throw_exception
    Kokkos::abort("Aborted in parallel_for, resilience majority voting failed because each execution obtained a differing value.");
    }

  } // execute
  inline ParallelFor(const FunctorType& arg_functor, Policy arg_policy)
      : m_instance(t_openmp_instance),
        m_functor(arg_functor),
        m_policy(arg_policy) {
    printf("Thread %d in resilient Kokkos, res pf constructor.\n", omp_get_thread_num());
  }
};// range policy implementation

} // namespace Impl
} // namespace Kokkos



#endif // KOKKOS_ENABLE_OPENMP //&& defined (KR_ENABLE_ACTIVE_EXECUTION_SPACE)
#endif // INC_RESILIENCE_OPENMP_OPENMPRESPARALLEL_HPP