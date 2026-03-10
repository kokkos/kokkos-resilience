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

#ifndef INC_RESILIENT_CUDA_SUBSCRIBER_HPP
#define INC_RESILIENT_CUDA_SUBSCRIBER_HPP
#include <Kokkos_Macros.hpp>
#if defined(KOKKOS_ENABLE_CUDA)

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <omp.h>
#include <iostream>
#include <cmath>
#include <map>
#include <typeinfo>
#include <unordered_map>
#include <sstream>
#include <utility>
#include <array>
#include <cstdint>
#include <cstddef>
#include <random>

#include <cuda_runtime.h>
#include <memory>

#include "Resilient_Cuda_Error_Injector.hpp"
#include "Resilient_Types.hpp"

/*--------------------------------------------------------------------------
 ******************** ERROR MESSAGE GENERATION *****************************
 --------------------------------------------------------------------------*/

namespace KokkosResilience {

  struct ResilientDuplicatesSubscriber;

  // Generate usable error message
  static_assert(Kokkos::Experimental::is_hooks_policy<
                Kokkos::Experimental::SubscribableViewHooks<
                ResilientDuplicatesSubscriber> >::value, "Must be a hooks policy");

}

/*----------------------------------------------------------------------------
 ******** STRUCT TO CHECK CORRECTNESS OF INDIVIDUAL ELEMENTS OF VIEWS ********
 ----------------------------------------------------------------------------*/

namespace KokkosResilience {

// Helper template used to get Rank number of 0's for MDRangePolicy
template< std::int64_t begin >
constexpr auto get_start_index ( std::size_t ){
  return begin;
}

// Makes a MDRangePolicy from View extents
template< typename View, std::size_t... Ranks >
auto make_md_range_policy( const View &view, std::index_sequence< Ranks... >){
  return Kokkos::MDRangePolicy<Kokkos::Rank<View::rank()>>({ 
    get_start_index < 0 > ( Ranks ) ... }, { static_cast<std::int64_t>( view.extent ( Ranks ))... });
}

// Checks on non-floating points, user can create own checker for custom structs
template <class Type, typename Enable=void>
struct CheckDuplicateEquality{

  KOKKOS_DEFAULTED_FUNCTION //Fixes nvcc warning issue, still on host and device
  CheckDuplicateEquality() = default;

  KOKKOS_DEFAULTED_FUNCTION
  CheckDuplicateEquality(const CheckDuplicateEquality& cde) = default;

  KOKKOS_INLINE_FUNCTION
  bool compare(Type a, Type b) const { return (a == b); }
};

// Checks equality of individual element on floating points
template <class Type>
struct CheckDuplicateEquality<
Type, typename std::enable_if< std::is_floating_point < Type >::value, void >::type > {

  KOKKOS_DEFAULTED_FUNCTION
  CheckDuplicateEquality() = default;

  KOKKOS_DEFAULTED_FUNCTION
  CheckDuplicateEquality(const CheckDuplicateEquality& cde) = default;

  KOKKOS_INLINE_FUNCTION
  bool compare(Type a, Type b) const { return (abs(a - b) < 0.00000001); }
};

} // namespace KokkosResilience

namespace KokkosResilience{

struct CombineDuplicatesBase
{
  // Virtual bool to return success flag
  virtual ~CombineDuplicatesBase() = default;
  virtual void clear() = 0;
  virtual bool execute() = 0;
  virtual void inject_error() = 0;
  // First index tracks first view, second index tracks second copy
  bool already_copied[2] = {false,false};

};

template< typename View >
struct CombineDuplicates: public CombineDuplicatesBase
{

  using EqualityType = CheckDuplicateEquality<typename View::value_type>;
  EqualityType check_equality;

  int duplicate_count = 0;
  View original;

  // Note: 2 copies allocated even in DMR
  View copy[2];

  bool success = 0;
  Kokkos::View <bool, Kokkos::Cuda> d_success{"device_success"};

  static constexpr size_t rank = View::rank();

#if defined KR_ERROR_INJECTION  
  void inject_error() override{
    error_injection(original, copy[0], copy[1]);
  }
#endif

  void clear() override
  {
    copy[0] = View ();
    copy[1] = View ();
  }

  bool execute() override
  {
    success = true;
    Kokkos::deep_copy(d_success, success);

#ifdef KR_DOUBLE_MODULAR_REDUNDANCY
    if (duplicate_count < 1){
    //if(already_copied[0]==false){
      Kokkos::abort("Aborted in CombineDuplicates, no duplicate created");
    }
#else
    if (duplicate_count < 2) {
    //if(already_copied[1]==false){
      Kokkos::abort("Aborted in CombineDuplicates, duplicate_count < 2");
    }
#endif

    else {

      //auto range_policy = Kokkos::RangePolicy<Kokkos::Cuda>(0,original.size());
      //Kokkos::parallel_for("Combiner", range_policy, *this);

      if constexpr(rank > 1){

        std::cout << "In multi-dimensional, rank = " << rank << std::endl;      
        auto mdrange = make_md_range_policy( original, std::make_index_sequence< rank > {} );
        Kokkos::parallel_for(mdrange, *this);
        std::cout << "#### After combine-duplicates parallel_for in the combiner map"<<std::endl;
  
      }else{

        std::cout << "original.size is " << original.size() << std::endl;
        Kokkos::parallel_for("SubscriberCombiner1D", original.size(), *this);
        std::cout << "#### After combine-duplicates parallel_for in the combiner map"<<std::endl;
  
      }
      //Kokkos::fence();
      // fence in deep_copy already
      Kokkos::deep_copy(success, d_success);
      return success;
    }
  }

  template <typename... Args>
  KOKKOS_INLINE_FUNCTION
  void operator ()(Args&&... vidx) const{
    //int tid = blockIdx.x * blockDim.x + threadIdx.x; 
    //Main combiner begin
    for (int j = 0; j < 2; j ++) {
      if (check_equality.compare(copy[j](std::forward<Args>(vidx)...), original(std::forward<Args>(vidx)...))) {
#if 0
	double valueo = static_cast<typename View::value_type>(original(std::forward<Args>(vidx)...));
        printf("O-C loop tid %i vidx %i original %f\n", tid, std::forward<Args>(vidx)..., valueo);
        double value0 = static_cast<typename View::value_type>(copy[0](std::forward<Args>(vidx)...));
        printf("O-C loop tid %i vidx %i copy[0] %f\n", tid, std::forward<Args>(vidx)..., value0);
        double value1 = static_cast<typename View::value_type>(copy[1](std::forward<Args>(vidx)...));
        printf("O-C loop tid %i vidx %i copy[1] %f\n", tid, std::forward<Args>(vidx)..., value1);      
#endif	
        return;
      }
    }
    if (check_equality.compare(copy[0](std::forward<Args>(vidx)...), copy[1](std::forward<Args>(vidx)...))) {
      original(std::forward<Args>(vidx)...) = static_cast<typename View::value_type>(copy[0](std::forward<Args>(vidx)...));        
#if 0
      double valueo = static_cast<typename View::value_type>(original(std::forward<Args>(vidx)...));
      printf("C-C loop tid %i vidx %i original %f\n", tid, std::forward<Args>(vidx)..., valueo);
      double value0 = static_cast<typename View::value_type>(copy[0](std::forward<Args>(vidx)...));
      printf("C-C loop tid %i vidx %i copy[0] %f\n", tid, std::forward<Args>(vidx)..., value0);
      double value1 = static_cast<typename View::value_type>(copy[1](std::forward<Args>(vidx)...));
      printf("C-C loop tid %i vidx %i copy[1] %f\n", tid, std::forward<Args>(vidx)..., value1);
#endif
      return;
    }
    //No match found, all three executions return different number
    d_success()=false;
//#if 0
    int tid = blockIdx.x * blockDim.x + threadIdx.x; 

    printf("tid %i vid %i errored with success = %i", tid, std::forward<Args>(vidx)..., d_success());
    double valueo = static_cast<typename View::value_type>(original(std::forward<Args>(vidx)...));
    printf("expecting error tid %i vidx %i original %f\n", tid, std::forward<Args>(vidx)..., valueo);
    double value0 = static_cast<typename View::value_type>(copy[0](std::forward<Args>(vidx)...));
    printf("expecting error tid %i vidx %i copy[0] %f\n", tid, std::forward<Args>(vidx)..., value0);
    double value1 = static_cast<typename View::value_type>(copy[1](std::forward<Args>(vidx)...));
    printf("expecting error tid %i vidx %i copy[1] %f\n", tid, std::forward<Args>(vidx)..., value1);
//#endif
  }

};// end Combiner

} // namespace KokkosResilience

/*----------------------------------------------------------------------------
 ************ SUBSCRIBER TO DUPLICATE VIEWS, COPY DATA ***********************
 ----------------------------------------------------------------------------*/

namespace KokkosResilience {

struct ResilientDuplicatesSubscriber {

  // Gating for using subscriber only inside resilient parallel loops
  static int resilient_duplicate_counter;
  //static bool in_resilient_parallel_loop;

 //  KOKKOS_IMPL_CUDA_SAFE_CALL

  // Creating map for duplicates: used for duplicate resolution per-kernel
  // Creating cache map of duplicates: used for tracking duplicates between kernels so that they are initialzied
  // only once. Re-initialize copies to be like original view only if original not in cache map
  using key_type = void *;  // key_type should be data() pointer
  inline static std::unordered_map<key_type, CombineDuplicatesBase *> duplicates_map;
  inline static std::unordered_map<key_type, std::unique_ptr<CombineDuplicatesBase> > duplicates_cache;

  template<typename View>
  static CombineDuplicates<View> *
  get_duplicate_for( const View &original) {
    bool inserted = false;
    auto pos = duplicates_cache.find(original.data());

    // True if got to end of cache and view wasn't found
    if (pos == duplicates_cache.end()) {
      // Insert view into cache map and flag
      inserted = true;
      pos = duplicates_cache.emplace(std::piecewise_construct,
                                     std::forward_as_tuple(original.data()),
                                     std::forward_as_tuple(
                                             std::make_unique<CombineDuplicates<View> >())).first;
    }

    auto &res = *static_cast< CombineDuplicates<View> * >( pos->second.get());
    // If inserted in the cache map then create copies and reinitialize

    //TODO: There may be some subtle issues with this in DMR since it only checks copy[0] and it won't trigger on the second copy, which would still need resizing. Consider a solution
    bool extents_resized = false;
    for (size_t i = 0; i < View::rank; ++i)
    {
      if (original.extent(i) != res.copy[0].extent(i) )
      {
        extents_resized = true;
        break;
      }
    }

    if (inserted || extents_resized) {
      res.original = original;

      // Reinitialize self to be like other (same dimensions, etc)
      for (int i = 0; i < 2; ++i) {
          set_duplicate_view(res.copy[i], original, i);
      }

    }
    return &res;
  }
  // Function which initializes the dimensions of the duplicating view
  template<typename View>
  static void set_duplicate_view(View &duplicate, const View &original, int duplicate_count) {
    std::stringstream label_ss;
    label_ss << original.label() << duplicate_count;
    
    auto err = cudaGetLastError();
    if (err != cudaSuccess)
      std::cout << "error:" << cudaGetErrorString(err);

    duplicate = View(Kokkos::view_alloc(label_ss.str(),Kokkos::WithoutInitializing), original.layout());
  }

  template<typename View>
  static void copy_constructed( View &self, const View &other)
  {
    if constexpr( std::is_same_v< typename View::non_const_data_type, typename View::data_type > )
    {
      // If view is non-constant and in the parallel loop, cascade the rest of the subscriber
      if (resilient_duplicate_counter > 0) {
	//if(in_resilient_parallel_loop){

        auto err = cudaGetLastError();
        if (err != cudaSuccess)
          std::cout << "error:" << cudaGetErrorString(err);

        // This won't be triggered if the entry already exists
        auto *combiner = get_duplicate_for(other);
        auto res = duplicates_map.emplace(std::piecewise_construct,
                                          std::forward_as_tuple(other.data()),
                                          std::forward_as_tuple(combiner));
        auto &c = dynamic_cast< CombineDuplicates< View > & > (*res.first->second);

        // The first copy constructor in a parallel_for for the given view
        if (res.second) {
	  //std::cout << "First entry to copy constructor for this view. Copying for state 1" << std::endl;	   assert(resilient_duplicate_counter == 1);
          c.duplicate_count = 0;
          //std::cout << "c_duplicate_count = "<<c.duplicate_count<< " and already_copied["<<c.duplicate_count<<"]="<<c.already_copied[c.duplicate_count]<<std::endl;
          c.already_copied[c.duplicate_count]=true;
          self = c.copy[c.duplicate_count++];
          //std::cout << "*****In copy constructor, self has now been set to copy c.duplicate_count=" << c.duplicate_count - 1 << "****" << std::endl;
	  Kokkos::deep_copy(self, other);
          //std::cout << "This is the end of the first-time constructor after Kokkos::deep_copy" <<std::endl;
          //std::cout << "c_duplicate_count = "<<c.duplicate_count<< " and already_copied["<<c.duplicate_count - 1 <<"]="<<c.already_copied[c.duplicate_count - 1]<<std::endl <<std::endl;
	}
	else if(resilient_duplicate_counter==2 & !c.already_copied[c.duplicate_count]){
          assert(resilient_duplicate_counter == 2);		
	  c.duplicate_count = 1;
	  //std::cout << "Now in the duplicate_count else branch" <<std::endl;
//	}

          //std::cout << "c_duplicate_count = "<<c.duplicate_count<< " and already_copied["<<c.duplicate_count<<"]="<<c.already_copied[c.duplicate_count]<<std::endl;
//        if (!c.already_copied[c.duplicate_count]){
          //self = c.copy[c.duplicate_count];
	
          c.already_copied[c.duplicate_count]=true;	
	  self = c.copy[c.duplicate_count++];
          //std::cout << "*****In copy constructor, self has now been set to copy c.duplicate_count=" << c.duplicate_count - 1 << "****" << std::endl;      
          //TODO: Check logic here
	  //c.already_copied[c.duplicate_count]=true;
          // Copy all data, every time
          Kokkos::deep_copy(self, other);
          //std::cout << "This is the end of the second constructor after Kokkos::deep_copy" <<std::endl;
	  //std::cout << "c_duplicate_count = "<<c.duplicate_count<< " and already_copied["<<c.duplicate_count - 1 <<"]="<<c.already_copied[c.duplicate_count - 1]<<std::endl <<std::endl;
        }

      }
    }
  }

  // Added to comply with new Subscriber format
  template <typename View>
  static void move_constructed(View &self, const View &other) {}

  template <typename View>
  static void move_assigned(View &self, const View &other) {}

  template <typename View>
  static void copy_assigned(View &self, const View &other) {}
};

//DO NOT mark KOKKOS_INLINE, marks function as device and host

inline void clear_duplicates_map() {
  for(auto&&[key,duplicate]:KokkosResilience::ResilientDuplicatesSubscriber::duplicates_map){
    duplicate->already_copied[0]=false;
    duplicate->already_copied[1]=false;
  }
  KokkosResilience::ResilientDuplicatesSubscriber::duplicates_map.clear();
}

inline void clear_duplicates_cache() {
  for (auto&& combiner : KokkosResilience::ResilientDuplicatesSubscriber::duplicates_cache) {
    combiner.second->clear();
  }
  KokkosResilience::ResilientDuplicatesSubscriber::duplicates_cache.clear();
}

} //namespace KokkosResilience

#endif //defined(KOKKOS_ENABLE_CUDA)
#endif //INC_RESILIENT_CUDA_SUBSCRIBER_HPP
