//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#ifndef RES_CUDA_PARALLEL_HPP
#define RES_CUDA_PARALLEL_HPP

#include <Kokkos_Macros.hpp>
#if defined(KR_ENABLE_CUDA)

#include <algorithm>
#include <string>
#include <typeinfo> 

#include <Kokkos_Core.hpp>

#include "ResCudaSubscriber.hpp"
 
/*--------------------------------------------------------------------------*/
/************************** DUPLICATE COMBINER CALL *************************/
/*--------------------------------------------------------------------------*/

namespace KokkosResilience{

  inline bool combine_resilient_duplicates() {

    bool success = true;
    // Combines all duplicates
    // Go over the Subscriber map, execute all the CombinerBase elements
    // If any duplicate fails to find a match, breaks
    for (auto&& combiner : KokkosResilience::CudaResilientSubscriber::duplicates_map) {
      success = combiner.second->execute();
      if(!success) break;
    }
    return success;
  }

} // namespace KokkosResilience

/*--------------------------------------------------------------------------*/
/************************ RESILIENT PARALLEL FORS ***************************/
/*--------------------------------------------------------------------------*/

// NOTE:: Remember KOKKOS_IMPL_DEBUG_CUDA_SERIAL_EXECUTION!!

namespace Kokkos {
namespace Impl {

// Range policy implementation
template <class FunctorType, class... Traits>
class ParallelFor< FunctorType
		 , Kokkos::RangePolicy<Traits...>
 		 , KokkosResilience::ResCuda> {
 public:
  using Policy = Kokkos::RangePolicy<Traits...>;
 private:
  using Member       = typename Policy::member_type;
  using WorkTag      = typename Policy::work_tag;
  using LaunchBounds = typename Policy::launch_bounds;

  const FunctorType m_functor;
  const Policy m_policy;

  ParallelFor() = delete;
  ParallelFor& operator=(const ParallelFor&) = delete;

  using surrogate_policy = Kokkos::RangePolicy < Kokkos::Cuda, WorkTag, LaunchBounds>;

 public:

  inline void execute() const {
    if (KokkosResilience::CudaResilientSubscriber::resilient_duplicate_counter > 0){
      Impl::ParallelFor< decltype(m_functor), Policy, Kokkos::Cuda > closure( m_functor ,m_policy );
 
    closure.execute(); 

    }
    else{ 

    surrogate_policy streamPolicy[3];
    for (int i = 0; i < 3; i++){
      cudaStream_t stream;
      cudaStreamCreate(&stream);
      Kokkos::Cuda cuda_inst(stream);
      new (&streamPolicy[i]) surrogate_policy(cuda_inst, m_policy.begin(), m_policy.end());
    }

    // This section after Subscriber, test with 1 tream
    KokkosResilience::CudaResilientSubscriber::resilient_duplicate_counter = 1;
    std::cout << "In parallel for, resilient_duplicate_counter = " 
    << KokkosResilience::CudaResilientSubscriber::resilient_duplicate_counter 
    << std::endl;
    auto m_functor_0 = m_functor;
    std::cout << "After first copy constructor, resilient_duplicate_counter = "        
    << KokkosResilience::CudaResilientSubscriber::resilient_duplicate_counter
    << std::endl;


    KokkosResilience::CudaResilientSubscriber::resilient_duplicate_counter = 2;
    std::cout << "After first copy constructor, resilient_duplicate_counter = " 
    << KokkosResilience::CudaResilientSubscriber::resilient_duplicate_counter 
    << std::endl;
    auto m_functor_1 = m_functor;
    std::cout << "After second copy constructor, resilient_duplicate_counter = "        
    << KokkosResilience::CudaResilientSubscriber::resilient_duplicate_counter
    << std::endl;
    KokkosResilience::CudaResilientSubscriber::resilient_duplicate_counter = 0;
    std::cout << "In parallel for, resilient_duplicate_counter = " 
    << KokkosResilience::CudaResilientSubscriber::resilient_duplicate_counter 
    << std::endl;

    
    Kokkos::fence();

    Impl::ParallelFor< decltype(m_functor), surrogate_policy, Kokkos::Cuda > closure0( m_functor , streamPolicy[0] );
    Impl::ParallelFor< decltype(m_functor), surrogate_policy, Kokkos::Cuda > closure1( m_functor_0 , streamPolicy[1] );
    Impl::ParallelFor< decltype(m_functor), surrogate_policy, Kokkos::Cuda > closure2( m_functor_1 , streamPolicy[2] );
     
    std::cout << "Before executes" << std::endl;
    
    closure0.execute();
    closure1.execute();
    closure2.execute();
    
    std::cout << "After executes" << std::endl;
    
  
    Kokkos::fence();
    
    std::cout << "Before combine " <<std::endl;
   
Kokkos::Profiling::pushRegion("CombinerCall");

    KokkosResilience::combine_resilient_duplicates();
    std::cout << "After combine " << std::endl;
Kokkos::Profiling::popRegion();
       
    std::cout<< "Before clear_duplicates_map" <<std::endl;
    KokkosResilience::clear_duplicates_map();
    std::cout<< "After clear_duplicates_map" <<std::endl;
  } 
  }
  ParallelFor(const FunctorType& arg_functor, 
	      const Policy& arg_policy)
            : m_functor(arg_functor), 
              m_policy(arg_policy) {}
};

} // namespace Impl
} // namespace Kokkos

#endif //(KR_ENABLE_CUDA)
#endif //RES_CUDA_PARALLEL_HPP

