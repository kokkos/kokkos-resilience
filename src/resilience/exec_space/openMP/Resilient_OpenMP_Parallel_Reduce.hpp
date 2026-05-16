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

#ifndef INC_RESILIENCE_OPENMP_PARALLELREDUCE_HPP
#define INC_RESILIENCE_OPENMP_PARALLELREDUCE_HPP

#include <Kokkos_Macros.hpp>
#include "resilience/ErrorHandler.hpp"

#if defined(KOKKOS_ENABLE_OPENMP)
#include <omp.h>
#include <iostream>

#include <Kokkos_Core.hpp>

#include "Resilient_OpenMP_Subscriber.hpp"
#include "Resilient_OpenMP_Error_Injector.hpp"
#include "Resilient_OpenMP_Duplicate_Map_Traversals.hpp"

namespace Kokkos {
namespace Impl {

template <class CombinedFunctorReducerType, class... Traits>
class ParallelReduce< CombinedFunctorReducerType
                    , Kokkos::RangePolicy< Traits... >
                    , KokkosResilience::ResOpenMP>{

 private:
  using Policy      = Kokkos::RangePolicy<Traits...>;
  using WorkTag     = typename Policy::work_tag;

  const CombinedFunctorReducerType m_functor_reducer;
  const Policy m_policy;
  using surrogate_policy = Kokkos::RangePolicy < Kokkos::OpenMP, WorkTag >;

  ParallelReduce() = delete;
  ParallelReduce & operator = ( const ParallelReduce & ) = delete;
  
  struct ReduceResultCombinerBase
  {

#ifdef KR_TRIPLE_MODULAR_REDUNDANCY	  
    // A pure virtual function, given a functor reference (from the constructor)
    virtual void execute_resilient_reduction( const CombinedFunctorReducerType &f
                                            , const CombinedFunctorReducerType &f0
                                            , const CombinedFunctorReducerType &f1
                                            , const surrogate_policy & pass_policy) = 0;
#endif

    virtual bool combine_reducers () = 0;
  };

  // Unique pointer to combinerbase struct
  std::unique_ptr<ReduceResultCombinerBase> m_combiner;

  template< typename ViewType >
  struct ReduceResultCombiner : ReduceResultCombinerBase
  {
    //create managed views to get duplicates for the reduction
    using ManagedViewType = Kokkos::View<typename ViewType::data_type
                            , typename ViewType::array_layout
                            , Kokkos::OpenMP>;

    static ManagedViewType create_managed_view(ViewType reducer_view, int duplicate_count) {

      std::stringstream label_ss;
      label_ss << reducer_view.label() << duplicate_count;
      return ManagedViewType(label_ss.str(), reducer_view.layout());

    }

    ReduceResultCombiner( ViewType reducer_view )
    {
      original = reducer_view;
      reducer_copy[0] = create_managed_view (reducer_view, 0);
      reducer_copy[1] = create_managed_view (reducer_view, 1);
    }

    ManagedViewType reducer_copy[2];
    ViewType original;

#ifdef KR_TRIPLE_MODULAR_REDUNDANCY
 
    void execute_resilient_reduction( const CombinedFunctorReducerType &f
                                    , const CombinedFunctorReducerType &f0
                                    , const CombinedFunctorReducerType &f1
                                    , const surrogate_policy & pass_policy) override
    {

      Impl::ParallelReduce< decltype(m_functor_reducer), surrogate_policy, Kokkos::OpenMP > closure1{ f, pass_policy, original };
      Impl::ParallelReduce< decltype(m_functor_reducer), surrogate_policy, Kokkos::OpenMP > closure2{ f0, pass_policy, reducer_copy[0] };
      Impl::ParallelReduce< decltype(m_functor_reducer), surrogate_policy, Kokkos::OpenMP > closure3{ f1, pass_policy, reducer_copy[1] };

      closure1.execute();
      closure2.execute();
      closure3.execute();

    }

#endif

    //resiliency on just the reducers
    bool combine_reducers () {

      using EqualityType = KokkosResilience::CheckDuplicateEquality<typename ViewType::value_type>;
      EqualityType check_equality;

      bool success = true;

      for (int j = 0; j < 2; j ++) {
        if (check_equality.compare(reducer_copy[j](), original())) {
          return success;
        }
      }
      if (check_equality.compare(reducer_copy[0](), reducer_copy[1]())) // just need 2 that are the same
        return success;
        //No match found, all three executions return different number
      success = false;
      return success;
    }
  };

 public:
  void execute() {

    bool success = 0; //! This bool indicates that all views successfully reached a consensus.
    surrogate_policy pass_policy;

#ifdef KR_TRIPLE_MODULAR_REDUNDANCY

    pass_policy = surrogate_policy(m_policy.begin(), m_policy.end());

    KokkosResilience::ResilientDuplicatesSubscriber::in_resilient_parallel_loop = true;
    auto m_functor_reducer_0 = m_functor_reducer;
    auto m_functor_reducer_1 = m_functor_reducer;
    KokkosResilience::ResilientDuplicatesSubscriber::in_resilient_parallel_loop = false;

    m_combiner->execute_resilient_reduction(m_functor_reducer, m_functor_reducer_0, m_functor_reducer_1, pass_policy);
    success = m_combiner->combine_reducers();
    if(!success){
      // Abort if no agreement in reduction
      auto &handler = KokkosResilience::get_unrecoverable_data_corruption_handler();
      handler(0);
    }

    success = KokkosResilience::combine_resilient_duplicates();

    // Does not clear the cache map, user must clear cache map before Kokkos::finalize()
    KokkosResilience::clear_duplicates_map();

#endif

    if(success==0){
      // Abort if no agreement in duplicates
      auto &handler = KokkosResilience::get_unrecoverable_data_corruption_handler();
      handler(0);
    }
  } //execute

  template < class ViewType >
  inline ParallelReduce( const CombinedFunctorReducerType& arg_functor_reducer
                       , Policy arg_policy
                       , const ViewType& arg_view)
                       : m_functor_reducer(arg_functor_reducer)
                       , m_policy (arg_policy)
  {
    m_combiner = std::make_unique<ReduceResultCombiner< ViewType > >(arg_view);
  }

};// range policy implementation

} //namespace Impl
} //namespace Kokkos

#endif // KOKKOS_ENABLE_OPENMP
#endif // INC_RESILIENCE_OPENMP_PARALLELREDUCE_HPP
