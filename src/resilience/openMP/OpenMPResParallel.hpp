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

// *****************************
// MDRange policy implementation
template <class FunctorType, class... Traits>
class ParallelFor< FunctorType
                  , Kokkos::MDRangePolicy<Traits...>
                  , KokkosResilience::ResOpenMP>{

 private:
  using MDRangePolicy = Kokkos::MDRangePolicy<Traits...>;
  using Policy        = typename MDRangePolicy::impl_range_policy;
  using WorkTag       = typename MDRangePolicy::work_tag;

  OpenMPExec* m_instance;
  const FunctorType &  m_functor;
  const MDRangePolicy m_mdr_policy;
  const Policy m_policy; // construct as RangePolicy( 0, num_tiles
                         // ).set_chunk_size(1) in ctor

  ParallelFor() = delete;
  ParallelFor & operator = ( const ParallelFor & ) = delete ;

  using surrogate_mdr_policy = Kokkos::MDRangePolicy < Kokkos::OpenMP, WorkTag >;
  //using surrogate_policy = MDRangePolicy::impl_range_policy;

 public:
  inline void execute() const {

    int repeats = 5; //! This integer represents the maximum number of attempts to reach consensus allowed.
    bool success = 0; //! This bool indicates that all views successfully reached a consensus.

    while(success==0 && repeats > 0){

      surrogate_mdr_policy wrapper_policy;
      wrapper_policy = surrogate_policy(m_policy.begin(), 3 * m_policy.end());

      // parallel_for turns off shared allocation tracking, toggle it back on for ViewHooks
      //Kokkos::Impl::shared_allocation_tracking_enable();

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
      };

      // TODO:
      // ALL THREAD SCHEDULING HANDLED BY KOKKOS HERE, ITERATION HANDLING BY US
      // Attempt to feed in a three-times as long range policy (wrapper-policy)
      // With a wrapped functor, so that the iterations are bound to the duplicated functors/views
      Impl::ParallelFor< decltype(wrapper_functor) , surrogate_mdr_policy, Kokkos::OpenMP > closure( wrapper_functor , wrapper_policy );

      // Execute it.
      closure.execute();

      Kokkos::fence();
      KokkosResilience::print_duplicates_map();
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
                     MDRangePolicy arg_policy)
      : m_instance(t_openmp_instance),
        m_functor(arg_functor),
        m_mdr_policy(arg_policy),
        m_policy(Policy(0, m_mdr_policy.m_num_tiles).set_chunk_size(1)) {}
  template <typename Policy, typename Functor>
  static int max_tile_size_product(const Policy&, const Functor&) {
    return 1024; // Unsure if this restriction needs to persist from MD Range in main Kokkos.
  }

};// MD range policy implementation


} // namespace Impl
} // namespace Kokkos

/*--------------------------------------------------------------------------*/
/*********************** RESILIENT PARALLEL REDUCES *************************/
/*--------------------------------------------------------------------------*/

namespace Kokkos {
namespace Impl {

// Range policy reduce
template<class FunctorType, class ReducerType, class... Traits>
class ParallelReduce< FunctorType
                    , Kokkos::RangePolicy< Traits... >
                    , ReducerType
                    , KokkosResilience::ResOpenMP> {
 private:
  using Policy    = Kokkos::RangePolicy<Traits...>;
  using WorkTag   = typename Policy::work_tag;
  using WorkRange = typename Policy::WorkRange;
  using Member    = typename Policy::member_type;

  using Analysis =
        FunctorAnalysis<FunctorPatternInterface::REDUCE, Policy, FunctorType>;

  using ReducerConditional =
        Kokkos::Impl::if_c<std::is_same<InvalidType, ReducerType>::value,
                           FunctorType, ReducerType>;

  using ReducerTypeFwd = typename ReducerConditional::type;
  using WorkTagFwd =
        std::conditional_t<std::is_same<InvalidType, ReducerType>::value, WorkTag,
                           void>;

  // Static Assert WorkTag void if ReducerType not InvalidType

  using ValueInit = Kokkos::Impl::FunctorValueInit<ReducerTypeFwd, WorkTagFwd>;
  using ValueJoin = Kokkos::Impl::FunctorValueJoin<ReducerTypeFwd, WorkTagFwd>;

  using pointer_type = typename Analysis::pointer_type;
  using reference_type = typename Analysis::reference_type;

  OpenMPExec * m_instance;
  const FunctorType m_functor;
  const Policy m_policy;
  const ReducerType m_reducer;
  const pointer_type m_result_ptr;

  ParallelReduce() = delete ;
  ParallelReduce & operator = ( const ParallelReduce & ) = delete ;

  using surrogate_policy = Kokkos::RangePolicy < Kokkos::OpenMP, WorkTag >;

 public:
  inline void execute() const {
    int repeats = 5;  //! This integer represents the maximum number of attempts to reach consensus allowed.
    bool success = 0;  //! This bool indicates that all views successfully reached a consensus.

    while (success == 0 && repeats > 0) {
      // TODO: SHOULD THERE BE A GUARD ON THE END SIZE? AND IF SO WHAT BEHAVIOR DESIRED?
      surrogate_policy wrapper_policy;
      wrapper_policy = surrogate_policy(m_policy.begin(), 3 * m_policy.end());

      // parallel_for turns off shared allocation tracking, toggle it back on for ViewHooks
      // Kokkos::Impl::shared_allocation_tracking_enable();

      // Trigger Subscriber constructors
      KokkosResilience::ResilientDuplicatesSubscriber::
          in_resilient_parallel_loop = true;
      auto m_functor_0               = m_functor;
      auto m_functor_1               = m_functor;
      auto m_functor_2               = m_functor;
      KokkosResilience::ResilientDuplicatesSubscriber::
          in_resilient_parallel_loop = false;

      auto wrapper_functor = [&](auto i) {
        if (i < m_policy.end()) {
          m_functor_0(i);
        } else if ((m_policy.end() <= i) && (i < 2 * m_policy.end())) {
          m_functor_1(i - m_policy.end());
        } else {
          m_functor_2(i - (2 * m_policy.end()));
        }
      };

      // Duplicate reducer-type and wrap
      auto m_reducer_0 = m_reducer;
      auto m_reducer_1 = m_reducer;
      auto m_reducer_2 = m_reducer;

      auto wrapper_reducer = [&](auto i) {
        if (i < m_policy.end()) {
          m_reducer_0;
        } else if ((m_policy.end() <= i) && (i < 2 * m_policy.end())) {
          m_reducer_1;
        } else {
          m_functor_2;
        }
      };

      // toggle the shared allocation tracking off again
      // Allows for user-intended view behavior in main body of parallel_for
      // Kokkos::Impl::shared_allocation_tracking_disable();

      // TODO:
      // ALL THREAD SCHEDULING HANDLED BY KOKKOS HERE, ITERATION HANDLING BY US
      // Attempt to feed in a three-times as long range policy (wrapper-policy)
      // With a wrapped functor, so that the iterations are bound to the duplicated functors/views
      Impl::ParallelReduce<decltype(wrapper_functor), surrogate_policy,
                           decltype(wrapper_reducer), Kokkos::OpenMP>
          closure(wrapper_functor, wrapper_policy);

      // Execute it.
      closure.execute();

      Kokkos::fence();
      KokkosResilience::print_duplicates_map();
      Kokkos::fence();

      // Combine the duplicate views and majority vote on correctness
      success = KokkosResilience::combine_resilient_duplicates();

      Kokkos::fence();

      KokkosResilience::clear_duplicates_map();
      repeats--;

    }  // while (!success & repeats left)

    if (success == 0 && repeats == 0) {
      // Abort if 5 repeated tries at executing failed to find acceptable match
      Kokkos::abort(
          "Aborted in parallel_for, resilience majority voting failed because each execution obtained a differing value.");
    }
  }
/*

                if (m_policy.end() <= m_policy.begin()) {
                    if (m_result_ptr) {
                        ValueInit::init(ReducerConditional::select(m_functor, m_reducer),
                                        m_result_ptr);
                        Kokkos::Impl::FunctorFinal<ReducerTypeFwd, WorkTagFwd>::final(
                                ReducerConditional::select(m_functor, m_reducer), m_result_ptr);
                    }
                    return;
                }

                enum {
                    is_dynamic = std::is_same<typename Policy::schedule_type::type,
                            Kokkos::Dynamic>::value
                };

                OpenMPExec::verify_is_master("Kokkos::OpenMP parallel_reduce");

                const size_t pool_reduce_bytes =
                        Analysis::value_size(ReducerConditional::select(m_functor, m_reducer));

                m_instance->resize_thread_data(pool_reduce_bytes, 0  // team_reduce_bytes
                        , 0  // team_shared_bytes
                        , 0  // thread_local_bytes
                );

                const int pool_size = OpenMP::impl_thread_pool_size();

#pragma omp parallel num_threads(pool_size)
                {
                    HostThreadTeamData &data = *(m_instance->get_thread_data());

                    data.set_work_partition(m_policy.end() - m_policy.begin(),
                                            m_policy.chunk_size());

                    if (is_dynamic) {
                        // Make sure work partition is set before stealing
                        if (data.pool_rendezvous()) data.pool_rendezvous_release();
                    }

                    reference_type update =
                            ValueInit::init(ReducerConditional::select(m_functor, m_reducer),
                                            data.pool_reduce_local());

                    std::pair<int64_t, int64_t> range(0, 0);

                    do {
                        range = is_dynamic ? data.get_work_stealing_chunk()
                                           : data.get_work_partition();

                        ParallelReduce::template exec_range<WorkTag>(
                                m_functor, range.first + m_policy.begin(),
                                range.second + m_policy.begin(), update);

                    } while (is_dynamic && 0 <= range.first);
                }

                // Reduction:

                const pointer_type ptr =
                        pointer_type(m_instance->get_thread_data(0)->pool_reduce_local());

                for (int i = 1; i < pool_size; ++i) {
                    ValueJoin::join(ReducerConditional::select(m_functor, m_reducer), ptr,
                                    m_instance->get_thread_data(i)->pool_reduce_local());
                }

                Kokkos::Impl::FunctorFinal<ReducerTypeFwd, WorkTagFwd>::final(
                        ReducerConditional::select(m_functor, m_reducer), ptr);

                if (m_result_ptr) {
                    const int n = Analysis::value_count(
                            ReducerConditional::select(m_functor, m_reducer));

                    for (int j = 0; j < n; ++j) {
                        m_result_ptr[j] = ptr[j];
                    }
                }
            }
*/
//----------------------------------------

  template<class ViewType>
  inline ParallelReduce( const FunctorType &arg_functor,
                                    Policy arg_policy,
                            const ViewType &arg_view,
                                  typename std::enable_if<Kokkos::is_view<ViewType>::value &&
                                                         !Kokkos::is_reducer_type<ReducerType>::value,
                                           void *>::type = nullptr)
                       : m_instance(t_openmp_instance),
                         m_functor(arg_functor),
                         m_policy(arg_policy),
                         m_reducer(InvalidType()),
                         m_result_ptr(arg_view.data()) {
                /*static_assert( std::is_same< typename ViewType::memory_space
                                                , Kokkos::HostSpace >::value
                  , "Reduction result on Kokkos::OpenMP must be a Kokkos::View in HostSpace"
                  );*/
  }

  inline ParallelReduce(const FunctorType &arg_functor,
                                   Policy arg_policy,
                                   const ReducerType &reducer)
                      : m_instance(t_openmp_instance),
                        m_functor(arg_functor),
                        m_policy(arg_policy),
                        m_reducer(reducer),
                        m_result_ptr(reducer.view().data()) {
                /*static_assert( std::is_same< typename ViewType::memory_space
                                                , Kokkos::HostSpace >::value
                  , "Reduction result on Kokkos::OpenMP must be a Kokkos::View in HostSpace"
                  );*/
  }
}; // range policy parallel reduce

} // namespace Impl
} // namespace Kokkos


#endif // KOKKOS_ENABLE_OPENMP
#endif // INC_RESILIENCE_OPENMP_OPENMPRESPARALLEL_HPP