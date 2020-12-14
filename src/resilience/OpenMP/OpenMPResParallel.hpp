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
#if defined(KOKKOS_ENABLE_OPENMP)// && defined (KR_LIZ_TODO) //&& defined (KR_ENABLE_ACTIVE_EXECUTION_SPACE)

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

namespace KokkosResilience {

  // Combine the resilient duplicates
  static void combine_res_duplicates() {
    printf("Entered combine_res_duplicates\n");
    fflush(stdout);
    std::map<std::string, KokkosResilience::DuplicateTracker* >::iterator it=ResHostSpace::duplicate_map.begin();
    while ( it != ResHostSpace::duplicate_map.end() ) {
      KokkosResilience::DuplicateTracker * dt = it->second;
      dt->combine_dups();
      it++;
    }
  }

  // Create a duplicate. Data record duplicated, original copied to the duplicate
  // and duplicte then assigned to the tracking element of the original. View
  // record points to duplicate.
  inline void duplicate_shared ( Kokkos::Experimental::ViewHolderBase &dst
                               , Kokkos::Experimental::ViewHolderBase &src ) {
    
    printf("This is Thread %d in resilient Kokkos. Entered duplicate_shared in ResOMPParallel.\n", omp_get_thread_num());
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

// Note namespace
namespace Kokkos {
namespace Impl {

//ALL THE OLD WORK
/*
// Range policy implementation
template< class FunctorType, class... Traits>
class ParallelFor< FunctorType
                 , Kokkos::RangePolicy< Traits... >
                 , KokkosResilience::ResOpenMP>
{
 private:
  
  typedef Kokkos::RangePolicy< Traits ... > Policy;
  typedef typename Policy::member_type         Member;
  typedef typename Policy::work_tag           WorkTag;
  typedef typename Policy::launch_bounds LaunchBounds;
 
  const FunctorType  & m_functor; // &m_functor, was m_functor in OpenMP
  const Policy         m_policy;

  // new items
  ParallelFor() = delete ;
  ParallelFor & operator = ( const ParallelFor & ) = delete;

 public:
  
  typedef FunctorType functor_type;

  inline void execute() const
  {
    typedef Kokkos::RangePolicy<Kokkos::OpenMP, WorkTag, LaunchBounds> surrogate_policy;
   
//TESTING PRINT STATEMENT
    printf("This is Thread %d in resilient Kokkos. Entered ParallelFor in ResOMPParallel, a typedefs, b surrogate.\n", omp_get_thread_num());
    fflush(stdout); 

    surrogate_policy lPolicy[3];
    for (int i = 0; i < 3; i++) {
      new (&lPolicy[i]) surrogate_policy(m_policy.begin(), m_policy.end());
    }

//TESTING PRINT STATEMENT
    printf("This is Thread %d in resilient Kokkos. Entered ParallelFor in ResOMPParallel, a surrogate, b ViewHooks.\n", omp_get_thread_num());
    fflush(stdout);     
    // ViewHooks captures non-constant views and passes to duplicate_shared
    Kokkos::Impl::shared_allocation_tracking_enable();

    auto vhc = Kokkos::Experimental::ViewHooks::create_view_hook_copy_caller(
                 []( Kokkos::Experimental::ViewHolderBase &dst
                   , Kokkos::Experimental::ViewHolderBase &src){
                       KokkosResilience::duplicate_shared(dst, src);
               });

    Kokkos::Experimental::ViewHooks::set("ResOpenMPDup", vhc); // May need to check ViewHooks

    // Resilient execution setups, should execute on different partitions
    // TODO: needs testing different partitions
    
//TESTING PRINT STATEMENT
    printf("This is Thread %d in resilient Kokkos. Entered ParallelFor in ResOMPParallel, a ViewHooks, b closure setup.\n", omp_get_thread_num());
    fflush(stdout); 

    Kokkos::Impl::ParallelFor< FunctorType, surrogate_policy, Kokkos::OpenMP> closureI( m_functor , lPolicy[0] );

    Kokkos::Impl::ParallelFor< FunctorType, surrogate_policy, Kokkos::OpenMP> closureII( m_functor , lPolicy[1] );

    Kokkos::Impl::ParallelFor< FunctorType, surrogate_policy, Kokkos::OpenMP> closureIII( m_functor , lPolicy[2] );

    Kokkos::Experimental::ViewHooks::clear("ResOpenMPDup", vhc);
    Kokkos::Impl::shared_allocation_tracking_disable(); //TODO: ???




// This is under testing to find the segmentation fault.
// Possible mod: partition
// Possible issue: omp parallel_for also uses impl_thread_pool_size(), conflict with task?
// Shrink thread pool size to 1/3 each?   

 
//TESTING PRINT STATEMENT
    printf("This is Thread %d in resilient Kokkos. Entered ParallelFor in ResOMPParallel, a closure setup, b execute.\n", omp_get_thread_num());
    fflush(stdout);


    #pragma omp parallel num_threads(OpenMP::impl_thread_pool_size())
    {
      
      //#pragma omp single 
      //{
        //Any thread the taskmaster thread
        #pragma omp task untied
        #pragma omp task
        { 
          //TESTING PRINT STATEMENT
          printf("This is Thread %d in resilient Kokkos. Entered OMP Task 1.\n", omp_get_thread_num());
          fflush(stdout);

          closureI.execute();
        }
       
        #pragma omp task
        {
          //TESTING PRINT STATEMENT
          printf("This is Thread %d in resilient Kokkos. Entered OMP Task 2.\n", omp_get_thread_num());
          fflush(stdout);
     
          closureII.execute();
        }

        #pragma omp task
        {
          //TESTING PRINT STATEMENT
          printf("This is Thread %d in resilient Kokkos. Entered OMP Task 3.\n", omp_get_thread_num());
          fflush(stdout);

          closureIII.execute();
        }

     //} //pragma omp single
    } //pragma omp parallel

    Kokkos::fence();

    KokkosResilience::combine_res_duplicates();

  }

  // No way this compiles - it didn't
  ParallelFor( const FunctorType &arg_functor, const Policy &arg_policy)
    : m_functor ( arg_functor )
    , m_policy ( arg_policy )
 {
   printf("This is Thread %d in resilient Kokkos. This is the res pf constructor.\n", omp_get_thread_num());
 }

}; // RangePolicy template ParallelFor
*/

// Range policy implementation NEW - ASK DAISY   
template <class FunctorType, class... Traits>
class ParallelFor< FunctorType
                 , Kokkos::RangePolicy< Traits... >
                 , KokkosResilience::ResOpenMP>{
 private:
  using Policy    = Kokkos::RangePolicy<Traits...>;
  using WorkTag   = typename Policy::work_tag;
  using WorkRange = typename Policy::WorkRange;
  using Member    = typename Policy::member_type;
  // Possible issue but should be fine to use WorkRange instead of LaunchBounds
  // using LaunchBounds = typename Policy::launch_bounds;

  OpenMPExec* m_instance;
  const FunctorType m_functor;
  const Policy m_policy;

  // templated class for range of memory? Used on policy.begin, policy.end later
  template <class TagType>
  inline static
      typename std::enable_if<std::is_same<TagType, void>::value>::type
      exec_range(const FunctorType& functor, const Member ibeg,
                 const Member iend) {

// Do I need to account for this at this time?
#ifdef KOKKOS_ENABLE_AGGRESSIVE_VECTORIZATION
#ifdef KOKKOS_ENABLE_PRAGMA_IVDEP
#pragma ivdep
#endif
#endif
    for (Member iwork = ibeg; iwork < iend; ++iwork) {
      functor(iwork);
    }
  }

  //I understand there is a difference, but not the subtlety of the difference
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

 public:
  inline void execute() const {
    //keep for now for printout similarity
    printf("This is Thread %d in resilient Kokkos range-policy ParallelFor, before parallel pragma.\n", omp_get_thread_num());
    fflush(stdout);

    enum {
      is_dynamic = std::is_same<typename Policy::schedule_type::type,
                                Kokkos::Dynamic>::value
    };

    typedef Kokkos::RangePolicy<Kokkos::OpenMP, WorkTag, WorkRange> surrogate_policy;
   
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
    
    // What does the if do, pretty sure it was meeting this condition with pre-port run.
    // Some work needed here
    if (OpenMP::in_parallel()) {
      exec_range<WorkTag>(m_functor, m_policy.begin(), m_policy.end());
    } else {
      OpenMPExec::verify_is_master("Kokkos::OpenMP parallel_for");

#pragma omp parallel num_threads(OpenMP::impl_thread_pool_size())
      {
        #pragma omp task
        {
          if(omp_get_thread_num() < floor(OpenMP::impl_thread_pool_size()/3)){  
         
             printf("This is Thread %d in resilient Kokkos range-policy ParallelFor, Task 1.\n", omp_get_thread_num());
             fflush(stdout);

             // Can I use the same instance, unduplicated?
             HostThreadTeamData& data = *(m_instance->get_thread_data());
 
             // Using surrogate policy duplicates
             data.set_work_partition(lPolicy[0].end() - lPolicy[0].begin(),
                                     lPolicy[0].chunk_size());

             if (is_dynamic) {
               // Make sure work partition is set before stealing
               if (data.pool_rendezvous()) data.pool_rendezvous_release();
             }
       
             std::pair<int64_t, int64_t> range(0, 0);

             do {
               range = is_dynamic ? data.get_work_stealing_chunk()
                                 : data.get_work_partition();
               ParallelFor::template exec_range<WorkTag>(
                           m_functor, range.first + lPolicy[0].begin(),
                           range.second + lPolicy[0].begin());
    
             } while (is_dynamic && 0 <= range.first);
           }
         }// omp task 1
           

        #pragma omp task
        {
          
	  if(omp_get_thread_num() > floor(OpenMP::impl_thread_pool_size())/3 &&
             omp_get_thread_num() < 2*(floor(OpenMP::impl_thread_pool_size())/3){  

             printf("This is Thread %d in resilient Kokkos range-policy ParallelFor, Task 2.\n", omp_get_thread_num());
             fflush(stdout);

             // Can I use the same instance, unduplicated?
             HostThreadTeamData& data = *(m_instance->get_thread_data());
 
             // Using surrogate policy duplicates
             data.set_work_partition(lPolicy[1].end() - lPolicy[1].begin(),
                                     lPolicy[1].chunk_size());

             if (is_dynamic) {
               // Make sure work partition is set before stealing
               if (data.pool_rendezvous()) data.pool_rendezvous_release();
             }
       
             std::pair<int64_t, int64_t> range(0, 0);

             do {
               range = is_dynamic ? data.get_work_stealing_chunk()
                                 : data.get_work_partition();
               ParallelFor::template exec_range<WorkTag>(
                           m_functor, range.first + lPolicy[1].begin(),
                           range.second + lPolicy[1].begin());
    
             } while (is_dynamic && 0 <= range.first);
           }
         }// omp task 2


        #pragma omp task
        {
	  if(omp_get_thread_num() <OpenMP::impl_thread_pool_size() &&
             omp_get_thread_num() > 2*(floor(OpenMP::impl_thread_pool_size())/3){  
         
             printf("This is Thread %d in resilient Kokkos range-policy ParallelFor, Task 3.\n", omp_get_thread_num());
             fflush(stdout);

             // Can I use the same instance, unduplicated?
             HostThreadTeamData& data = *(m_instance->get_thread_data());
 
             // Using surrogate policy duplicates
             data.set_work_partition(lPolicy[2].end() - lPolicy[2].begin(),
                                     lPolicy[2].chunk_size());

             if (is_dynamic) {
               // Make sure work partition is set before stealing
               if (data.pool_rendezvous()) data.pool_rendezvous_release();
             }
       
             std::pair<int64_t, int64_t> range(0, 0);

             do {
               range = is_dynamic ? data.get_work_stealing_chunk()
                                 : data.get_work_partition();
               ParallelFor::template exec_range<WorkTag>(
                           m_functor, range.first + lPolicy[2].begin(),
                           range.second + lPolicy[2].begin());
    
             } while (is_dynamic && 0 <= range.first);
           }
         }// omp task 3

      } // pragma omp
    } // omp-parallel else

    Kokkos::fence();
    
    // Correct place for teardown?
    Kokkos::Experimental::ViewHooks::clear("ResOpenMPDup", vhc);
    Kokkos::Impl::shared_allocation_tracking_disable(); 
       
    // The last thing before the constructor
    KokkosResilience::combine_res_duplicates();

  } // execute

  inline ParallelFor(const FunctorType& arg_functor, Policy arg_policy)
      : m_instance(t_openmp_instance),
        m_functor(arg_functor),
        m_policy(arg_policy) {
    printf("This is Thread %d in resilient Kokkos. This is the new version res pf constructor.\n", omp_get_thread_num());
  }
};//range policy implementation

} // namespace Impl
} // namespace Kokkos

/*--------------------------------------------------------------------------*/
/*********************** RESILIENT PARALLEL SCANS ***************************/
/*--------------------------------------------------------------------------*/
/*
// Note namespace
namespace Kokkos {
namespace Impl {

// Range policy implementation
template< class FunctorType, class... Traits>
class ParallelScan< FunctorType
                 , Kokkos::RangePolicy< Traits... >
                 , KokkosResilience::ResOpenMP>
{
 private:
  
  typedef Kokkos::RangePolicy< Traits ... > Policy;
  typedef typename Policy::member_type         Member;
  typedef typename Policy::work_tag           WorkTag;
  typedef typename Policy::launch_bounds LaunchBounds;
 
  const FunctorType  & m_functor;
  const Policy         m_policy;

  // new items
  ParallelScan() = delete ;
  ParallelScan & operator = ( const ParallelScan & ) = delete;

 public:
  
  typedef FunctorType functor_type;

  inline void execute() const
  {
    typedef Kokkos::RangePolicy<Kokkos::OpenMP, WorkTag, LaunchBounds> surrogate_policy;
   
//TESTING PRINT STATEMENT
    printf("Entered ParallelScan execute. Right after typedef, right before surrogate.\n");
    fflush(stdout); 

    surrogate_policy lPolicy[3];
    for (int i = 0; i < 3; i++) {
      new (&lPolicy[i]) surrogate_policy(m_policy.begin(), m_policy.end());
    }

//TESTING PRINT STATEMENT
    printf("Entered ParallelScan execute. Right after surrogate, right before ViewHooks.\n");
    fflush(stdout);     
    // ViewHooks captures non-constant views and passes to duplicate_shared
    Kokkos::Impl::shared_allocation_tracking_enable();

    auto vhc = Kokkos::Experimental::ViewHooks::create_view_hook_copy_caller(
                 []( Kokkos::Experimental::ViewHolderBase &dst
                   , Kokkos::Experimental::ViewHolderBase &src){
                       KokkosResilience::duplicate_shared(dst, src);
               });

    Kokkos::Experimental::ViewHooks::set("ResOpenMPDup", vhc); // May need to check ViewHooks

    // Resilient execution setups, should execute on different partitions
    // TODO: needs testing different partitions
    
//TESTING PRINT STATEMENT
    printf("Entered ParallelScan execute. Right after ViewHooks, right before closure setup.\n");
    fflush(stdout); 

    Kokkos::Impl::ParallelScan< FunctorType, surrogate_policy, Kokkos::OpenMP> closureI( m_functor , lPolicy[0] );

    Kokkos::Impl::ParallelScan< FunctorType, surrogate_policy, Kokkos::OpenMP> closureII( m_functor , lPolicy[1] );

    Kokkos::Impl::ParallelScan< FunctorType, surrogate_policy, Kokkos::OpenMP> closureIII( m_functor , lPolicy[2] );

    Kokkos::Experimental::ViewHooks::clear("ResOpenMPDup", vhc);
    Kokkos::Impl::shared_allocation_tracking_disable();
 
//TESTING PRINT STATEMENT
    printf("Entered ParallelScan execute. Right after closure setup, right before closure execute.\n");
    fflush(stdout); 

    #pragma omp parallel num_threads(OpenMP::impl_thread_pool_size())
    {
      
      #pragma omp single 
      {
        //Any thread the taskmaster thread
        #pragma omp task untied

        #pragma omp task
        { 
          //TESTING PRINT STATEMENT
          printf("We got into the 1st OMP task.\n");
          fflush(stdout);

          closureI.execute();
        }
       
        #pragma omp task
        {
          //TESTING PRINT STATEMENT
          printf("We got into the 2nd OMP task.\n");
          fflush(stdout);
     
          closureII.execute();
        }

        #pragma omp task
        {
          //TESTING PRINT STATEMENT
          printf("We got into the 3rd OMP task.\n");
          fflush(stdout);

          closureIII.execute();
        }

      } //pragma omp single
    } //pragma omp parallel


    Kokkos::fence();

    KokkosResilience::combine_res_duplicates();

  }

  // Resilient policy functor constructor
  ParallelScan( const FunctorType &arg_functor, const Policy &arg_policy)
    : m_functor ( arg_functor )
    , m_policy ( arg_policy )
 {
   printf("res pf constructor\n");
 }

}; // RangePolicy template ParallelScan

} // namespace Impl
} // namespace Kokkos
*/
/*--------------------------------------------------------------------------*/
/********************** RESILIENT PARALLEL REDUCES **************************/
/*--------------------------------------------------------------------------*/
/*
// Note namespace
namespace Kokkos {
namespace Impl {

// Range policy implementation
template< class FunctorType, class ReducerType, class... Traits>
class ParallelReduce< FunctorType
                 , Kokkos::RangePolicy< Traits... >
                 , ReducerType
                 , KokkosResilience::ResOpenMP>
{
 private:
  
  typedef Kokkos::RangePolicy< Traits ... > Policy;
  typedef typename Policy::member_type         Member;
  typedef typename Policy::work_tag           WorkTag;
  typedef typename Policy::launch_bounds LaunchBounds;
 
  const FunctorType  & m_functor;
  const Policy         m_policy;

  // Reduce specific
  const ReducerType & m_reducer;
  const pointer_type m_result_ptr;
  const pointer_type m_result_ptr_1;
  const pointer_type m_result_ptr_2;

  // new items
  ParallelReduce() = delete ;
  ParallelReduce & operator = ( const ParallelReduce & ) = delete;

 public:
  
  typedef FunctorType functor_type;

  inline void execute() const
  {
    typedef Kokkos::RangePolicy<Kokkos::OpenMP, WorkTag, LaunchBounds> surrogate_policy;
   
//TESTING PRINT STATEMENT
    printf("Entered ParallelReduce execute. Right after typedef, right before surrogate.\n");
    fflush(stdout); 

    surrogate_policy lPolicy[3];
    for (int i = 0; i < 3; i++) {
      new (&lPolicy[i]) surrogate_policy(m_policy.begin(), m_policy.end());
    }

//TESTING PRINT STATEMENT
    printf("Entered ParallelReduce execute. Right after surrogate, right before ViewHooks.\n");
    fflush(stdout);     
    // ViewHooks captures non-constant views and passes to duplicate_shared
    Kokkos::Impl::shared_allocation_tracking_enable();

    auto vhc = Kokkos::Experimental::ViewHooks::create_view_hook_copy_caller(
                 []( Kokkos::Experimental::ViewHolderBase &dst
                   , Kokkos::Experimental::ViewHolderBase &src){
                       KokkosResilience::duplicate_shared(dst, src);
               });

    Kokkos::Experimental::ViewHooks::set("ResOpenMPDup", vhc); // May need to check ViewHooks

    // Resilient execution setups, should execute on different partitions
    // TODO: needs testing different partitions
    
//TESTING PRINT STATEMENT
    printf("Entered ParallelReduce execute. Right after ViewHooks, right before closure setup.\n");
    fflush(stdout); 

    Kokkos::Impl::ParallelReduce< FunctorType, ReducerType, surrogate_policy, 
                                  Kokkos::OpenMP> closureI( m_functor, m_reducer, lPolicy[0] );

    Kokkos::Impl::ParallelReduce< FunctorType, ReducerType, surrogate_policy, 
                                  Kokkos::OpenMP> closureII( m_functor, m_reducer, lPolicy[1] );

    Kokkos::Impl::ParallelReduce< FunctorType, ReducerType. surrogate_policy, 
                                  Kokkos::OpenMP> closureIII( m_functor, m_reducer, lPolicy[2] );

    Kokkos::Experimental::ViewHooks::clear("ResOpenMPDup", vhc);
    Kokkos::Impl::shared_allocation_tracking_disable();
 
//TESTING PRINT STATEMENT
    printf("Entered ParallelReduce execute. Right after closure setup, right before closure execute.\n");
    fflush(stdout); 

    #pragma omp parallel num_threads(OpenMP::impl_thread_pool_size())
    {
      
      #pragma omp single 
      {
        //Any thread the taskmaster thread
        #pragma omp task untied

        #pragma omp task
        { 
          //TESTING PRINT STATEMENT
          printf("We got into the 1st OMP task.\n");
          fflush(stdout);

          closureI.execute();
        }
       
        #pragma omp task
        {
          //TESTING PRINT STATEMENT
          printf("We got into the 2nd OMP task.\n");
          fflush(stdout);
     
          closureII.execute();
        }

        #pragma omp task
        {
          //TESTING PRINT STATEMENT
          printf("We got into the 3rd OMP task.\n");
          fflush(stdout);

          closureIII.execute();
        }

      } //pragma omp single
    } //pragma omp parallel


    Kokkos::fence();

    KokkosResilience::combine_res_duplicates();

  }

//May not be neccessary prepare to delete, try first with just the result pointer stuff

  template <class ViewType>
  inline ParallelReduce(
      const FunctorType& arg_functor, Policy arg_policy,
      const ViewType& arg_view,
      typename std::enable_if<Kokkos::is_view<ViewType>::value &&
                                  !Kokkos::is_reducer_type<ReducerType>::value,
                              void*>::type = nullptr)
      :  m_functor(arg_functor)
      ,  m_policy(arg_policy)
      ,  m_reducer(InvalidType())
      ,  m_result_ptr(arg_view.data()) 
  {
    printf("Template needed after all!\n");  
  }


  // Resilient policy functor constructor
  ParallelReduce( const FunctorType &arg_functor, const Policy &arg_policy,
                  const ReducerType &reducer)
    : m_functor ( arg_functor )
    , m_policy ( arg_policy )

    , m_result_ptr(reducer.view().data())
 {
   printf("res pf constructor\n");
 }

}; // RangePolicy template ParallelReduce

} // namespace Impl
} // namespace Kokkos
*/
/*--------------------------------------------------------------------------*/

#endif // KOKKOS_ENABLE_OPENMP //&& defined (KR_ENABLE_ACTIVE_EXECUTION_SPACE)
#endif // INC_RESILIENCE_OPENMP_OPENMPRESPARALLEL_HPP

