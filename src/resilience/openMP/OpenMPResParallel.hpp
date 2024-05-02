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
#include "resilience/Resilience.hpp"
#if defined(KOKKOS_ENABLE_OPENMP)

#include <omp.h>
#include <iostream>

#include <Kokkos_Core.hpp>

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

  inline void inject_error_duplicates() {

    if (KokkosResilience::global_error_settings->error_rate){
      //Per kernel, seed the first inject index	    
      KokkosResilience::ErrorInject::global_next_inject = KokkosResilience::global_error_settings->geometric(KokkosResilience::ErrorInject::random_gen);	      ;

      // Inject geometrically distributed error into all duplicates
      // Go over the Subscriber map, inject into all the CombinerBase elements
      for (auto&& combiner : KokkosResilience::ResilientDuplicatesSubscriber::duplicates_map) {
        combiner.second->inject_error();
      }
    }
    else{
      std::cout << "Error finding error_rate. Were global error settings enabled?\n";
    }
  }

} // namespace KokkosResilience



/*--------------------------------------------------------------------------*/
/************************* ERROR INSERTION TEST CODE ************************/
/*--------------------------------------------------------------------------*/
/*
namespace KokkosResilience{

inline insert_error_duplicates(){
// Goes over the Subscriber duplicate map (not cache map), inserting an error randomly
  for (auto&& combiner : KokkosResilience::ResilientDuplicatesSubscriber::duplicates_map) {
    combiner.second->insert_error();
  }
}

} // namespace KokkosResilience
*/
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
//MiniMD
  using LaunchBounds = typename Policy::launch_bounds;
  using Member       = typename Policy::member_type;

  const FunctorType &  m_functor;
  const Policy m_policy;

  ParallelFor() = delete ;
  ParallelFor & operator = ( const ParallelFor & ) = delete ;

  using surrogate_policy = Kokkos::RangePolicy < Kokkos::OpenMP, WorkTag, LaunchBounds>;

 public:
  inline void execute() const {
    //! The execution() function in this class performs an OpenMP execution of parallel for with
    //! triple modular redundancy. Non-constant views equipped with the triggering subscribers are
    //! duplicated and three concurrent executions divided equally between the available pool
    //! of OpenMP threads proceed. Duplicate views are combined back into a single view by calling
    //! a combiner to majority vote on the correct values. This process is repeated until
    //! a value is voted correct or a given number of attempts is exceeded.
    //! There are some subtleties regarding which views are copied per kernel in the default subscriber
    //! See KokkosResilience::ResilienctDuplicatesSubscriber::duplicates_cache for details

    const int max_repeats = 5;
    int repeats = max_repeats; //! This integer represents the maximum number of attempts to reach consensus allowed.
    bool success = 0; //! This bool indicates that all views successfully reached a consensus.

    while(success==0 && repeats > 0){
      surrogate_policy wrapper_policy;

#ifdef KR_ENABLE_TMR
      wrapper_policy = surrogate_policy(m_policy.begin(), m_policy.end());
      // Trigger Subscriber constructors
      KokkosResilience::ResilientDuplicatesSubscriber::in_resilient_parallel_loop = true;
      auto m_functor_0 = m_functor;
      auto m_functor_1 = m_functor;
      KokkosResilience::ResilientDuplicatesSubscriber::in_resilient_parallel_loop = false;
#endif

#ifdef KR_ENABLE_WRAPPER
      auto work_size = m_policy.end() - m_policy.begin();
      auto offset = m_policy.begin();
      wrapper_policy = surrogate_policy(0, 3 * work_size );

      // Trigger Subscriber constructors
      KokkosResilience::ResilientDuplicatesSubscriber::in_resilient_parallel_loop = true;
      auto m_functor_0 = m_functor;
      auto m_functor_1 = m_functor;
      KokkosResilience::ResilientDuplicatesSubscriber::in_resilient_parallel_loop = false;

      auto wrapper_functor = [&](auto i){
        if (i < work_size)
        {
          m_functor (i + offset);
        }
        else if (( work_size <= i) && (i < (2 * work_size)))
        {
          m_functor_0 (i + offset - work_size);
        }
        else
        {
          m_functor_1 (i + offset - ( 2 * work_size));
        }
      };
#endif

#ifdef KR_ENABLE_TMR

      // TMR execution with no wrapper scheduling

      Impl::ParallelFor< decltype(m_functor) , surrogate_policy, Kokkos::OpenMP > closure0( m_functor , wrapper_policy );
      Impl::ParallelFor< decltype(m_functor) , surrogate_policy, Kokkos::OpenMP > closure1( m_functor_0 , wrapper_policy );
      Impl::ParallelFor< decltype(m_functor) , surrogate_policy, Kokkos::OpenMP > closure2( m_functor_1 , wrapper_policy );

      closure0.execute();
      closure1.execute();
      closure2.execute();


      const auto start{std::chrono::steady_clock::now()};
      KokkosResilience::inject_error_duplicates();
      const auto stop{std::chrono::steady_clock::now()};
      KokkosResilience::ETimer::elapsed_seconds = KokkosResilience::ETimer::elapsed_seconds + (std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start));
      KokkosResilience::ETimer::total_error_time = KokkosResilience::ETimer::total_error_time + KokkosResilience::ETimer::elapsed_seconds;

      // Combine the duplicate views and majority vote on correctness
      success = KokkosResilience::combine_resilient_duplicates();
      //KokkosResilience::print_duplicates_map();
      // Does not clear the cache map, user must clear cache map before Kokkos::finalize()
      KokkosResilience::clear_duplicates_map();

#endif

#ifdef KR_ENABLE_DMR

      //DMR with failover to TMR on error
      wrapper_policy = surrogate_policy(m_policy.begin(), m_policy.end());

      // Trigger Subscriber constructors
      KokkosResilience::ResilientDuplicatesSubscriber::in_resilient_parallel_loop = true;
      auto m_functor_0 = m_functor;
      KokkosResilience::ResilientDuplicatesSubscriber::in_resilient_parallel_loop = false;

      // DMR execution with no wrapper scheduling, on failover TMR
      Impl::ParallelFor< decltype(m_functor) , surrogate_policy, Kokkos::OpenMP > closure0(m_functor , wrapper_policy );
      Impl::ParallelFor< decltype(m_functor) , surrogate_policy, Kokkos::OpenMP > closure1(m_functor_0 , wrapper_policy );

      closure0.execute();
      closure1.execute();

      const auto start{std::chrono::steady_clock::now()};
      KokkosResilience::inject_error_duplicates();
      const auto stop{std::chrono::steady_clock::now()};
      KokkosResilience::ETimer::elapsed_seconds = KokkosResilience::ETimer::elapsed_seconds + (std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start));
      KokkosResilience::ETimer::total_error_time = KokkosResilience::ETimer::total_error_time + KokkosResilience::ETimer::elapsed_seconds;
      
      // Combine the duplicate views, majority vote not triggered due to CMAKE macro
      success = KokkosResilience::combine_resilient_duplicates();

      if (!success)
      {
        KokkosResilience::ResilientDuplicatesSubscriber::dmr_failover_to_tmr = true;
        KokkosResilience::ResilientDuplicatesSubscriber::in_resilient_parallel_loop = true;
        auto m_functor_1 = m_functor;
        KokkosResilience::ResilientDuplicatesSubscriber::in_resilient_parallel_loop = false;

        Impl::ParallelFor< decltype(m_functor) , surrogate_policy, Kokkos::OpenMP > closure2(m_functor_1 , wrapper_policy );

        const auto start{std::chrono::steady_clock::now()};
        KokkosResilience::inject_error_duplicates();
        const auto stop{std::chrono::steady_clock::now()};
        KokkosResilience::ETimer::elapsed_seconds = KokkosResilience::ETimer::elapsed_seconds + (std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start));
        KokkosResilience::ETimer::total_error_time = KokkosResilience::ETimer::total_error_time + KokkosResilience::ETimer::elapsed_seconds;

        success = KokkosResilience::combine_resilient_duplicates();
        KokkosResilience::clear_duplicates_map();
        KokkosResilience::ResilientDuplicatesSubscriber::dmr_failover_to_tmr = false;
      }

      KokkosResilience::clear_duplicates_map();
#endif
#ifdef KR_ENABLE_WRAPPER

      // TMR with scheduling
      // Feed in three-times as long range policy (wrapper-policy)
      // With wrapped functor, so that the iterations are bound to the duplicated functors/views
      Impl::ParallelFor< decltype(wrapper_functor) , surrogate_policy, Kokkos::OpenMP > closure( wrapper_functor , wrapper_policy );

      closure.execute();

      const auto start{std::chrono::steady_clock::now()};
      KokkosResilience::inject_error_duplicates();
      const auto stop{std::chrono::steady_clock::now()};
      KokkosResilience::ETimer::elapsed_seconds = KokkosResilience::ETimer::elapsed_seconds + (std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start));
      KokkosResilience::ETimer::total_error_time = KokkosResilience::ETimer::total_error_time + KokkosResilience::ETimer::elapsed_seconds;

      // Combine the duplicate views and majority vote on correctness
      success = KokkosResilience::combine_resilient_duplicates();
      // Does not clear the cache map, user must clear cache map before Kokkos::finalize()
      KokkosResilience::clear_duplicates_map();
#endif

      repeats--;

    }// while (!success & repeats left)

    if(success==0 && repeats == 0){
      // Abort if 5 repeKokkos::abort(ated tries at executing failed to find acceptable match
      auto &handler = KokkosResilience::get_unrecoverable_data_corruption_handler();
      handler(max_repeats);
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

namespace Kokkos {
namespace Impl {

// MDRange policy implementation
template <class FunctorType, class... Traits>
class ParallelFor<FunctorType, 
		  Kokkos::MDRangePolicy<Traits...>,
                  KokkosResilience::ResOpenMP> {
 private:
  using MDRangePolicy = Kokkos::MDRangePolicy<Traits...>;
  using Policy        = typename MDRangePolicy::impl_range_policy;
  using WorkTag       = typename MDRangePolicy::work_tag;

  //TODO: MiniMD may need Launchbounds

  using Member = typename Policy::member_type;

  using index_type   = typename Policy::index_type;
  using iterate_type = typename Kokkos::Impl::HostIterateTile<
      MDRangePolicy, FunctorType, typename MDRangePolicy::work_tag, void>;

  const iterate_type m_iter;

  const FunctorType &  m_functor;
  const MDRangePolicy m_policy;

  ParallelFor() = delete ;
  ParallelFor & operator = ( const ParallelFor & ) = delete ;
  
  // TODO: MiniMD may require added LaunchBounds
  using surrogate_policy = Kokkos::MDRangePolicy < Kokkos::OpenMP, WorkTag>;

 public:
  inline void execute() const {

    if (KokkosResilience::ResOpenMP::in_parallel())
      Kokkos::abort("Cannot call resilient parallel_for inside a parallel construct.");

    int repeats = 5;
    bool success = 0;
    while(success==0 && repeats > 0){
  
      surrogate_policy wrapper_policy;
      wrapper_policy = surrogate_policy(m_policy.begin(), m_policy.end());
  
      // Trigger Subscriber constructors
      KokkosResilience::ResilientDuplicatesSubscriber::in_resilient_parallel_loop = true;
      auto m_functor_0 = m_functor;
      auto m_functor_1 = m_functor;
      KokkosResilience::ResilientDuplicatesSubscriber::in_resilient_parallel_loop = false;

      Impl::ParallelFor< decltype(m_functor) , surrogate_policy, Kokkos::OpenMP > closure0(m_functor , wrapper_policy );
      Impl::ParallelFor< decltype(m_functor) , surrogate_policy, Kokkos::OpenMP > closure1(m_functor_0 , wrapper_policy );
      Impl::ParallelFor< decltype(m_functor) , surrogate_policy, Kokkos::OpenMP > closure2(m_functor_1 , wrapper_policy );

      closure0.execute();
      closure1.execute();
      closure2.execute();

      // Combine the duplicate views and majority vote on correctness
      success = KokkosResilience::combine_resilient_duplicates();    
      KokkosResilience::clear_duplicates_map();
  
      repeats--;
  
      }// while (!success & repeats left)

    if(success==0 && repeats == 0){
      Kokkos::abort("Aborted in parallel_for, resilience majority voting failed because each execution obtained a differing value.");
    }

  } //execute

  inline ParallelFor(const FunctorType& arg_functor, MDRangePolicy arg_policy)
      : m_iter(arg_policy, arg_functor) {}
}; // MDRange policy impl

} // namespace Impl
} // namespace Kokkos

/*--------------------------------------------------------------------------*/
/*********************** RESILIENT PARALLEL REDUCES *************************/
/*--------------------------------------------------------------------------*/

namespace Kokkos {
namespace Impl {

//TODO:: This comment invalidated by rework of Kokkos with CombinedFunctorReducerType??
// Will eventually need enable_if to partially specialize on the different reducer types
// This specialization is for view type only, but written as only instantiation for now

// Range policy implementation
#if 0
template <class FunctorType, class ReducerType, class... Traits>
class ParallelReduce< FunctorType
                    , Kokkos::RangePolicy< Traits... >
                    , ReducerType
                    , KokkosResilience::ResOpenMP>{
#endif

template <class CombinedFunctorReducerType, class... Traits>
class ParallelReduce< CombinedFunctorReducerType
		    , Kokkos::RangePolicy< Traits... >
		    , KokkosResilience::ResOpenMP>{

 private:
  using Policy      = Kokkos::RangePolicy<Traits...>;
  //using FunctorType = typename CombinedFunctorReducerType::functor_type;
  using WorkTag     = typename Policy::work_tag;
  //TODO: may need to access reducer_view and may need in unieque pointer
  //using ReducerType = typename CombinedFuctorReducerType::reducor_type; 

  const CombinedFunctorReducerType m_functor_reducer;
 //const ReducerType& reducer = m_functor_reducer.get_reducer();
  //FunctorType m_functor;
  const Policy m_policy;
  using surrogate_policy = Kokkos::RangePolicy < Kokkos::OpenMP, WorkTag >;

  ParallelReduce() = delete;
  ParallelReduce & operator = ( const ParallelReduce & ) = delete;

  //primary template
  struct ReduceResultCombinerBase
  {
    // A pure virtual function, given a functor reference (from the constructor)
    virtual void execute_resilient_reduction( const CombinedFunctorReducerType &f
                                            , const CombinedFunctorReducerType &f0
                                            , const CombinedFunctorReducerType &f1
                                            , const surrogate_policy & pass_policy) = 0;

    virtual bool combine_reducers () = 0;
    virtual void print() = 0;
  };

  // Unique pointer to combinerbase struct
  std::unique_ptr<ReduceResultCombinerBase> m_combiner;

  //templated on ViewType, will override void function with triplicate execution for ViewTypes
  template< typename ViewType >
  struct ReduceResultCombiner : ReduceResultCombinerBase
  {
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
      reducer_copy[0] = create_managed_view (reducer_view, 1);
      reducer_copy[1] = create_managed_view (reducer_view, 2);
    }
	
    ManagedViewType reducer_copy[2];
    ViewType original;
/*
#ifdef KR_ENABLE_DMR
//TODO::Impl::ParallelReduce< FunctorType, surrogate_policy, InvalidType, Kokkos::OpenMP > closure1{ f, pass_policy, original };
//Original call with FunctorType, and ReducerType of InvalidType, no longer valid. FunctorType now represents just the functor of combinedreducerfunctor.

    void execute_resilient_reduction( const FunctorType &f
                                    , const FunctorType &f0
                                    , const FunctorType &f1
                                    , const surrogate_policy & pass_policy) override
    {

      Impl::ParallelReduce< FunctorType, surrogate_policy, Kokkos::OpenMP > closure1{ f, pass_policy, original };
      Impl::ParallelReduce< FunctorType, surrogate_policy, Kokkos::OpenMP > closure2{ f0, pass_policy, reducer_copy[0] };

      closure1.execute();
      closure2.execute();

    }

#else
*/
    void execute_resilient_reduction( const CombinedFunctorReducerType &f
                                    , const CombinedFunctorReducerType &f0
                                    , const CombinedFunctorReducerType &f1
                                    , const surrogate_policy & pass_policy) override
    {

      Impl::ParallelReduce< CombinedFunctorReducerType, surrogate_policy, Kokkos::OpenMP > closure1{ f, pass_policy, original };
      Impl::ParallelReduce< CombinedFunctorReducerType, surrogate_policy, Kokkos::OpenMP > closure2{ f0, pass_policy, reducer_copy[0] };
      Impl::ParallelReduce< CombinedFunctorReducerType, surrogate_policy, Kokkos::OpenMP > closure3{ f1, pass_policy, reducer_copy[1] };

      closure1.execute();
      closure2.execute();
      closure3.execute();

    }

//#endif

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

    void print() override {

      std::cout << std::endl;
      std::cout << original() << std::endl;
      std::cout << reducer_copy[0]() << std::endl;
      std::cout << reducer_copy[1]() << std::endl;
      std::cout << std::endl;

    }
  };

 public:
  void execute() {

    int repeats = 5; //! This integer represents the maximum number of attempts to reach consensus allowed.
    bool success = 0; //! This bool indicates that all views successfully reached a consensus.

    while(success==0 && repeats > 0){
/*
#if KR_ENABLE_DMR

      surrogate_policy pass_policy;
      pass_policy = surrogate_policy(m_policy.begin(), m_policy.end());

      KokkosResilience::ResilientDuplicatesSubscriber::in_resilient_parallel_loop = true;
      auto m_functor_0 = m_functor;
      KokkosResilience::ResilientDuplicatesSubscriber::in_resilient_parallel_loop = false;

      m_combiner->execute_resilient_reduction(m_functor, m_functor_0, m_functor_1, pass_policy);
      m_combiner->print();

      success = KokkosResilience::combine_resilient_duplicates();
 
// NO FAILOVER LOOP YET, STRAIGHT DMR -> logic issue in the reducer copy

#else
*/ 
      surrogate_policy pass_policy;
      pass_policy = surrogate_policy(m_policy.begin(), m_policy.end());

      KokkosResilience::ResilientDuplicatesSubscriber::in_resilient_parallel_loop = true;
      auto m_functor_reducer_0 = m_functor_reducer;
      auto m_functor_reducer_1 = m_functor_reducer;
      KokkosResilience::ResilientDuplicatesSubscriber::in_resilient_parallel_loop = false;

      m_combiner->execute_resilient_reduction(m_functor_reducer, m_functor_reducer_0, m_functor_reducer_1, pass_policy);
      //m_combiner->print();

      //KokkosResilience::print_duplicates_map();
      // Combine the duplicate views and majority vote on correctness
      success = KokkosResilience::combine_resilient_duplicates();

      // Does not clear the cache map, user must clear cache map before Kokkos::finalize()
      KokkosResilience::clear_duplicates_map();
      repeats--;

//#endif

    }// while (!success & repeats left)

    if(success==0 && repeats == 0){
      // Abort if 5 repeated tries at executing failed to find acceptable match
      Kokkos::abort("Aborted in resilient parallel_reduce.");
    }
  } //execute
#if 0
  template< typename ViewType >
  inline ParallelReduce(const FunctorType &f
                , Policy arg_policy
                , const ViewType &view_arg)
                : m_functor( f )
                , m_policy(arg_policy)
  {
    m_combiner = std::make_unique<ReduceResultCombiner< ViewType > >( view_arg );
  }
# endif

  template < class ViewType >
  inline ParallelReduce( const CombinedFunctorReducerType& arg_functor_reducer
 		       , Policy arg_policy
		       , const ViewType& arg_view)
                       : m_functor_reducer(arg_functor_reducer)
                       , m_policy (arg_policy)
  {
    m_combiner = std::make_unique<ReduceResultCombiner< ViewType > >(arg_view);
  }


/*
  template <class ViewType>
  inline ParallelReduce(const FunctorType& arg_functor,
                        Policy arg_policy,
                        const ViewType& arg_view,
                        typename std::enable_if<Kokkos::is_view<ViewType>::value &&
                                      !Kokkos::is_reducer_type<ReducerType>::value, void*>::type = nullptr)
                      : m_instance(t_openmp_instance),
                        m_functor(arg_functor),
                        m_policy(arg_policy),
                        m_reducer(arg_functor)
                        //,m_result_ptr(arg_view.data())
                        {
  }

  inline ParallelReduce(const FunctorType& arg_functor,
                        Policy arg_policy,
                        const ReducerType& reducer)
                        : m_instance(t_openmp_instance),
                          m_functor(arg_functor),
                          m_policy(arg_policy),
                          m_reducer(reducer),
                          m_result_ptr(reducer.view().data()) {
  }*/

};// range policy implementation

} // namespace Impl
} // namespace Kokkos

#endif // KOKKOS_ENABLE_OPENMP
#endif // INC_RESILIENCE_OPENMP_OPENMPRESPARALLEL_HPP
