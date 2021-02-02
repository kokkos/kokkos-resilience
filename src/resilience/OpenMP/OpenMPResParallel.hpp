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
#include <impl/TrackDuplicates.hpp>
//#include <impl/ViewHookSpecialization.hpp>

/*--------------------------------------------------------------------------*/

// Number of repeats before fail-out if all duplicate views in resilience
// return a different value.
//int repeats = 5;

/*--------------------------------------------------------------------------*/

namespace KokkosResilience {

  // Combine the resilient duplicates
  bool combine_res_duplicates() {
    printf("Entered combine_res_duplicates\n");
    fflush(stdout);
 /*   
 *  std::map<std::string, KokkosResilience::DuplicateTracker* >::iterator it=ResHostSpace::duplicate_map.begin();
    while ( it != ResHostSpace::duplicate_map.end() ) {
      KokkosResilience::DuplicateTracker * dt = it->second;
      dt->combine_dups();
      it++;
   }
*/
    

    bool success = true;
    for(auto& kv : ResHostSpace::duplicate_map) {
      success = kv.second->combine_dups();
      if(!success) break;
    }
    return success;

  }

  // Create a duplicate. Data record duplicated, original copied to the duplicate
  // and duplicte then assigned to the tracking element of the original. View
  // record points to duplicate.
  inline void duplicate_shared ( Kokkos::Experimental::ViewHolderBase &dst
                               , Kokkos::Experimental::ViewHolderBase &src ) {
    
    printf("Thread %d in resilient Kokkos. Entered duplicate_shared in ResOMPParallel.\n", omp_get_thread_num());
    fflush(stdout);
    // Assign new record to view map
    dst.update_view (src.rec_ptr() );

    // Copy data
    dst.deep_copy_from_buffer( (unsigned char *)src.data() );
  
  }

} //namespace KokkosResilience

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
  const FunctorType m_functor;
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
	     surrogate_policy& lPolicy,
             bool dynamic_value) const {

    printf("Thread %d in resilient Kokkos range-policy, executing work.\n", omp_get_thread_num());
    fflush(stdout);

    HostThreadTeamData& data = *(m_instance->get_thread_data());
 
    data.set_work_partition(lPolicy.end() - lPolicy.begin(),
                                     lPolicy.chunk_size());

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

    surrogate_policy lPolicy[3];
    for (int i = 0; i < 3; i++) {
      lPolicy[i] = surrogate_policy(m_policy.begin(), m_policy.end());      
    }
   
    // ViewHooks captures non-constant views and passes to duplicate_shared
    Kokkos::Impl::shared_allocation_tracking_enable();

    auto vhc = Kokkos::Experimental::ViewHooks::create_view_hook_copy_caller(
                 []( Kokkos::Experimental::ViewHolderBase &dst
                   , Kokkos::Experimental::ViewHolderBase &src){
                       KokkosResilience::duplicate_shared(dst, src);
               });

    Kokkos::Experimental::ViewHooks::set("ResOpenMPDup", vhc);
    
    if (OpenMP::in_parallel()) {
      exec_range<WorkTag>(m_functor, m_policy.begin(), m_policy.end());
    } else {
      OpenMPExec::verify_is_master("Kokkos::OpenMP parallel_for");

#pragma omp parallel num_threads(OpenMP::impl_thread_pool_size())
      {
        if(omp_get_thread_num() < OpenMP::impl_thread_pool_size()/3){  
      
          exec_work(m_functor, lPolicy[0], is_dynamic);
        }

        if(omp_get_thread_num() >= OpenMP::impl_thread_pool_size()/3 &&
           omp_get_thread_num() < (2*OpenMP::impl_thread_pool_size())/3){  

          exec_work(m_functor, lPolicy[1], is_dynamic);
        }

        if(omp_get_thread_num() < OpenMP::impl_thread_pool_size() &&
           omp_get_thread_num() >= (2*OpenMP::impl_thread_pool_size())/3){  
      
          exec_work(m_functor, lPolicy[2], is_dynamic);
        }
  
      } // pragma omp
    } // omp-parallel else

    Kokkos::fence();
    
    // Teardown
    Kokkos::Experimental::ViewHooks::clear("ResOpenMPDup", vhc);
    Kokkos::Impl::shared_allocation_tracking_disable(); 
       
    success = KokkosResilience::combine_res_duplicates();
    repeats--;

}// while (!success & repeats left)

if(success==0 && repeats == 0){
// TODO: Improve error message to give for label
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

/*--------------------------------------------------------------------------*/
/*********************** RESILIENT PARALLEL SCANS ***************************/
/*--------------------------------------------------------------------------*/

/*namespace Kokkos {
namespace Impl {

template <class FunctorType, class... Traits>
class ParallelScan< FunctorType, 
		    Kokkos::RangePolicy<Traits...>,
                    KokkosResilience::ResOpenMP> {
 private:
  using Policy = Kokkos::RangePolicy<Traits...>;

  using Analysis =
      FunctorAnalysis<FunctorPatternInterface::SCAN, Policy, FunctorType>;

  using WorkTag   = typename Policy::work_tag;
  using WorkRange = typename Policy::WorkRange;
  using Member    = typename Policy::member_type;

  using ValueInit = Kokkos::Impl::FunctorValueInit<FunctorType, WorkTag>;
  using ValueJoin = Kokkos::Impl::FunctorValueJoin<FunctorType, WorkTag>;
  using ValueOps  = Kokkos::Impl::FunctorValueOps<FunctorType, WorkTag>;

  using pointer_type   = typename Analysis::pointer_type;
  using reference_type = typename Analysis::reference_type;

  typedef Kokkos::RangePolicy<Kokkos::OpenMP, WorkTag, WorkRange> surrogate_policy;
   
  OpenMPExec* m_instance;
  const FunctorType m_functor;
  const Policy m_policy;

  template <class TagType>
  inline static
      typename std::enable_if<std::is_same<TagType, void>::value>::type
      exec_range(const FunctorType& functor, const Member ibeg,
                 const Member iend, reference_type update, const bool final) {
    for (Member iwork = ibeg; iwork < iend; ++iwork) {
      functor(iwork, update, final);
    }
  }

  template <class TagType>
  inline static
      typename std::enable_if<!std::is_same<TagType, void>::value>::type
      exec_range(const FunctorType& functor, const Member ibeg,
                 const Member iend, reference_type update, const bool final) {
    const TagType t{};
    for (Member iwork = ibeg; iwork < iend; ++iwork) {
      functor(t, iwork, update, final);
    }
  }

 public:
  inline void execute() const {
    OpenMPExec::verify_is_master("Kokkos::OpenMP parallel_scan");

    const int value_count          = Analysis::value_count(m_functor);
    const size_t pool_reduce_bytes = 2 * Analysis::value_size(m_functor);

    m_instance->resize_thread_data(pool_reduce_bytes, 0  // team_reduce_bytes
                                   ,
                                   0  // team_shared_bytes
                                   ,
                                   0  // thread_local_bytes
    );

    printf("This is Thread %d in resilient Kokkos range-policy ParallelScan, before parallel pragma.\n", omp_get_thread_num());
    fflush(stdout);
   
    surrogate_policy lPolicy[3];
    for (int i = 0; i < 3; i++) {
      lPolicy[i] = surrogate_policy(m_policy.begin(), m_policy.end());      
    }
   
    // ViewHooks captures non-constant views and passes to duplicate_shared
    Kokkos::Impl::shared_allocation_tracking_enable();

    auto vhc = Kokkos::Experimental::ViewHooks::create_view_hook_copy_caller(
                 []( Kokkos::Experimental::ViewHolderBase &dst
                   , Kokkos::Experimental::ViewHolderBase &src){
                       KokkosResilience::duplicate_shared(dst, src);
               });

    Kokkos::Experimental::ViewHooks::set("ResOpenMPDup", vhc);

#pragma omp parallel num_threads(OpenMP::impl_thread_pool_size())
    {
      
      if(omp_get_thread_num() < OpenMP::impl_thread_pool_size()/3){  
      
        HostThreadTeamData& data = *(m_instance->get_thread_data());

        const WorkRange range(lPolicy[0], omp_get_thread_num(),
                              omp_get_num_threads());

        reference_type update_sum =
            ValueInit::init(m_functor, data.pool_reduce_local());

        ParallelScan::template exec_range<WorkTag>(
            m_functor, range.begin(), range.end(), update_sum, false);

        if (data.pool_rendezvous()) {
          pointer_type ptr_prev = nullptr;

          const int n = omp_get_num_threads();

          for (int i = 0; i < n; ++i) {
            pointer_type ptr =
                (pointer_type)data.pool_member(i)->pool_reduce_local();

            if (i) {
              for (int j = 0; j < value_count; ++j) {
                ptr[j + value_count] = ptr_prev[j + value_count];
              }
              ValueJoin::join(m_functor, ptr + value_count, ptr_prev);
            } else {
              ValueInit::init(m_functor, ptr + value_count);
            }

            ptr_prev = ptr;
          }

          data.pool_rendezvous_release();
        }

        reference_type update_base = ValueOps::reference(
            ((pointer_type)data.pool_reduce_local()) + value_count);

        ParallelScan::template exec_range<WorkTag>(
            m_functor, range.begin(), range.end(), update_base, true);

        }

      if(omp_get_thread_num() >= OpenMP::impl_thread_pool_size()/3 &&
           omp_get_thread_num() < (2*OpenMP::impl_thread_pool_size())/3){  

      HostThreadTeamData& data = *(m_instance->get_thread_data());

      const WorkRange range(lPolicy[1], omp_get_thread_num(),
                            omp_get_num_threads());

      reference_type update_sum =
          ValueInit::init(m_functor, data.pool_reduce_local());

      ParallelScan::template exec_range<WorkTag>(
          m_functor, range.begin(), range.end(), update_sum, false);

      if (data.pool_rendezvous()) {
        pointer_type ptr_prev = nullptr;

        const int n = omp_get_num_threads();

        for (int i = 0; i < n; ++i) {
          pointer_type ptr =
              (pointer_type)data.pool_member(i)->pool_reduce_local();

          if (i) {
            for (int j = 0; j < value_count; ++j) {
              ptr[j + value_count] = ptr_prev[j + value_count];
            }
            ValueJoin::join(m_functor, ptr + value_count, ptr_prev);
          } else {
            ValueInit::init(m_functor, ptr + value_count);
          }

          ptr_prev = ptr;
        }

        data.pool_rendezvous_release();
      }

      reference_type update_base = ValueOps::reference(
          ((pointer_type)data.pool_reduce_local()) + value_count);

      ParallelScan::template exec_range<WorkTag>(
          m_functor, range.begin(), range.end(), update_base, true);



      }

      if(omp_get_thread_num() < OpenMP::impl_thread_pool_size() &&
           omp_get_thread_num() >= (2*OpenMP::impl_thread_pool_size())/3){  

      HostThreadTeamData& data = *(m_instance->get_thread_data());

      const WorkRange range(lPolicy[2], omp_get_thread_num(),
                            omp_get_num_threads());

      reference_type update_sum =
          ValueInit::init(m_functor, data.pool_reduce_local());

      ParallelScan::template exec_range<WorkTag>(
          m_functor, range.begin(), range.end(), update_sum, false);

      if (data.pool_rendezvous()) {
        pointer_type ptr_prev = nullptr;

        const int n = omp_get_num_threads();

        for (int i = 0; i < n; ++i) {
          pointer_type ptr =
              (pointer_type)data.pool_member(i)->pool_reduce_local();

          if (i) {
            for (int j = 0; j < value_count; ++j) {
              ptr[j + value_count] = ptr_prev[j + value_count];
            }
            ValueJoin::join(m_functor, ptr + value_count, ptr_prev);
          } else {
            ValueInit::init(m_functor, ptr + value_count);
          }

          ptr_prev = ptr;
        }

        data.pool_rendezvous_release();
      }

      reference_type update_base = ValueOps::reference(
          ((pointer_type)data.pool_reduce_local()) + value_count);

      ParallelScan::template exec_range<WorkTag>(
          m_functor, range.begin(), range.end(), update_base, true);

      
      }


    } // pragma omp

    Kokkos::fence();

    Kokkos::Experimental::ViewHooks::clear("ResOpenMPDup", vhc);
    Kokkos::Impl::shared_allocation_tracking_disable(); 
       
    KokkosResilience::combine_res_duplicates();

  } // execute

  //----------------------------------------

  inline ParallelScan(const FunctorType& arg_functor, const Policy& arg_policy)
        : m_instance(t_openmp_instance),
        m_functor(arg_functor),
        m_policy(arg_policy) {

    
    printf("This is Thread %d in resilient Kokkos, new PScan constructor.\n", omp_get_thread_num());
 
  }

  //----------------------------------------
};
 

} //Impl


}//Kokkos

*/



/*--------------------------------------------------------------------------*/
/********************** RESILIENT PARALLEL REDUCES **************************/
/*--------------------------------------------------------------------------*/

namespace Kokkos {
namespace Impl {

template <class FunctorType, class ReducerType, class... Traits>
class ParallelReduce<FunctorType, 
		     Kokkos::RangePolicy<Traits...>, 
		     ReducerType,
                     KokkosResilience::ResOpenMP> {
private:
  using Policy = Kokkos::RangePolicy<Traits...>;

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
      typename Kokkos::Impl::if_c<std::is_same<InvalidType, ReducerType>::value,
                                  WorkTag, void>::type;

  // Static Assert WorkTag void if ReducerType not InvalidType
  using ValueInit = Kokkos::Impl::FunctorValueInit<ReducerTypeFwd, WorkTagFwd>;
  using ValueJoin = Kokkos::Impl::FunctorValueJoin<ReducerTypeFwd, WorkTagFwd>;

  using pointer_type   = typename Analysis::pointer_type;
  using reference_type = typename Analysis::reference_type;

  typedef Kokkos::RangePolicy<Kokkos::OpenMP, WorkTag, WorkRange> surrogate_policy;
   
  OpenMPExec* m_instance;
  const FunctorType m_functor;
  const Policy m_policy;
  const ReducerType m_reducer;
  const pointer_type m_result_ptr;

  template <class TagType>
  inline static
      typename std::enable_if<std::is_same<TagType, void>::value>::type
      exec_range(const FunctorType& functor, const Member ibeg,
                 const Member iend, reference_type update) {
    for (Member iwork = ibeg; iwork < iend; ++iwork) {
      functor(iwork, update);
    }
  }

  template <class TagType>
  inline static
      typename std::enable_if<!std::is_same<TagType, void>::value>::type
      exec_range(const FunctorType& functor, const Member ibeg,
                 const Member iend, reference_type update) {
    const TagType t{};
    for (Member iwork = ibeg; iwork < iend; ++iwork) {
      functor(t, iwork, update);
    }
  }


///FEED IN REDUCER, SEE ABOUT UPDATE
  inline void
  exec_work (const FunctorType& functor,
             const ReducerType& reducer,
             surrogate_policy& lPolicy,
             bool is_dynamic) const {
    
    printf("Thread %d in resilient Kokkos rp-reduce, executing work.\n", omp_get_thread_num());
    fflush(stdout);

    HostThreadTeamData& data = *(m_instance->get_thread_data());

    data.set_work_partition(lPolicy.end() - lPolicy.begin(),
                              lPolicy.chunk_size());

    if (is_dynamic) {
    // Make sure work partition is set before stealing
      if (data.pool_rendezvous()) data.pool_rendezvous_release();
    }
    
    reference_type update = ValueInit::init(ReducerConditional::select(functor, reducer),
                                            data.pool_reduce_local());
    
    std::pair<int64_t, int64_t> range(0, 0);
    
    do {
      range = is_dynamic ? data.get_work_stealing_chunk()
                         : data.get_work_partition();
      ParallelReduce::template exec_range<WorkTag>(
                      functor, range.first + lPolicy.begin(),
                      range.second + lPolicy.begin(), update);
     
    } while (is_dynamic && 0 <= range.first);
  } 

public:
  inline void execute() const {
    if (m_policy.end() <= m_policy.begin()) {
      if (m_result_ptr) {
        ValueInit::init(ReducerConditional::select(m_functor, m_reducer),
                        m_result_ptr);
            Kokkos::Impl::FunctorFinal<ReducerTypeFwd, WorkTagFwd>::final(
      	            ReducerConditional::select(m_functor, m_reducer), m_result_ptr);
      }
      return;
    }

    printf("This is Thread %d in resilient Kokkos range-policy ParallelReduce, before parallel pragma.\n", omp_get_thread_num());
    fflush(stdout);

    static constexpr bool is_dynamic = std::is_same<typename Policy::schedule_type::type,
                                Kokkos::Dynamic>::value;

    surrogate_policy lPolicy[3];
    for (int i = 0; i < 3; i++) {
      lPolicy[i] = surrogate_policy(m_policy.begin(), m_policy.end());
    }

    // ViewHooks captures non-constant views and passes to duplicate_shared
    Kokkos::Impl::shared_allocation_tracking_enable();

    auto vhc = Kokkos::Experimental::ViewHooks::create_view_hook_copy_caller(
                     []( Kokkos::Experimental::ViewHolderBase &dst
                       , Kokkos::Experimental::ViewHolderBase &src){
                           KokkosResilience::duplicate_shared(dst, src);
      });

    Kokkos::Experimental::ViewHooks::set("ResOpenMPDup", vhc);

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
      
      if(omp_get_thread_num() < OpenMP::impl_thread_pool_size()/3){
 
        exec_work(m_functor, m_reducer, lPolicy[0], is_dynamic);   
       }

       if(omp_get_thread_num() >= OpenMP::impl_thread_pool_size()/3 &&
          omp_get_thread_num() < (2*OpenMP::impl_thread_pool_size())/3){

         exec_work(m_functor, m_reducer, lPolicy[1], is_dynamic);
       }

       if(omp_get_thread_num() < OpenMP::impl_thread_pool_size() &&
          omp_get_thread_num() >= (2*OpenMP::impl_thread_pool_size())/3){

         exec_work(m_functor, m_reducer, lPolicy[2], is_dynamic);
       }

    } //pragma omp
    
    Kokkos::fence();
    /*
    // Viewhooks teardown doesn't go here, will have to duplicate reduction
    Kokkos::Experimental::ViewHooks::clear("ResOpenMPDup", vhc);
    Kokkos::Impl::shared_allocation_tracking_disable(); 
       
    KokkosResilience::combine_res_duplicates();
    */
   

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

    Kokkos::Experimental::ViewHooks::clear("ResOpenMPDup", vhc);
    Kokkos::Impl::shared_allocation_tracking_disable(); 
       
    KokkosResilience::combine_res_duplicates();

  } // execute

  //----------------------------------------
  
  template <class ViewType>
  inline ParallelReduce(
      const FunctorType& arg_functor, Policy arg_policy,
      const ViewType& arg_view,
      typename std::enable_if<Kokkos::is_view<ViewType>::value &&
                                  !Kokkos::is_reducer_type<ReducerType>::value,
                              void*>::type = nullptr)
      : m_instance(t_openmp_instance),
        m_functor(arg_functor),
        m_policy(arg_policy),
        m_reducer(InvalidType()),
        m_result_ptr(arg_view.data()) {
  }

  inline ParallelReduce(const FunctorType& arg_functor, Policy arg_policy,
                        const ReducerType& reducer)
      : m_instance(t_openmp_instance),
        m_functor(arg_functor),
        m_policy(arg_policy),
        m_reducer(reducer),
        m_result_ptr(reducer.view().data()) {
  }
}; // resilient RangePolicy reduce

} // namespace Impl
} // namespace Kokkos

/*--------------------------------------------------------------------------*/

#endif // KOKKOS_ENABLE_OPENMP //&& defined (KR_ENABLE_ACTIVE_EXECUTION_SPACE)
#endif // INC_RESILIENCE_OPENMP_OPENMPRESPARALLEL_HPP

