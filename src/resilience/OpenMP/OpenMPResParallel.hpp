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
    
    printf("Entered duplicate_shared\n");
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
    printf("Entered ParallelFor execute. Right after typedef, right before surrogate.\n");
    fflush(stdout); 

    surrogate_policy lPolicy[3];
    for (int i = 0; i < 3; i++) {
      new (&lPolicy[i]) surrogate_policy(m_policy.begin(), m_policy.end());
    }

//TESTING PRINT STATEMENT
    printf("Entered ParallelFor execute. Right after surrogate, right before ViewHooks.\n");
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
    printf("Entered ParallelFor execute. Right after ViewHooks, right before closure setup.\n");
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
    printf("Entered ParallelFor execute. Right after closure setup, right before closure execute.\n");
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
   printf("res pf constructor\n");
 }

}; // RangePolicy template ParallelFor

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

