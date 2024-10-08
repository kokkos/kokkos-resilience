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

#ifndef INC_RESILIENCE_OPENMP_OPENMPRESSUBSCRIBER_HPP
#define INC_RESILIENCE_OPENMP_OPENMPRESSUBSCRIBER_HPP

#include <Kokkos_Macros.hpp>
#if defined(KOKKOS_ENABLE_OPENMP)

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
#include <random>

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

// Struct to gate error insertion
struct Error{

explicit Error(double rate) : error_rate(rate), geometric(rate){}

double error_rate;
std::geometric_distribution<> geometric{error_rate};

};

inline std::optional<Error> global_error_settings;

struct ErrorInject{
inline static int64_t error_counter;
inline static std::mt19937 random_gen{0};
inline static size_t global_next_inject = 0;
};

struct ETimer{
//Error timer settings zero-initialized
inline static std::chrono::duration<long int, std::ratio<1, 1000000000>> elapsed_seconds{};
inline static std::chrono::duration<long int, std::ratio<1, 1000000000>> total_error_time{};
inline static std::mutex global_time_mutex;
};

// Helper template used to get Rank number of 0's for MDRangePolicy
template< std::int64_t begin >
constexpr auto get_start_index ( std::size_t ){
  return begin;
}

// Makes a MDRangePolicy from View extents
template< typename View, std::size_t... Ranks >
auto make_md_range_policy( const View &view, std::index_sequence< Ranks... >){
  return Kokkos::MDRangePolicy<Kokkos::Rank<View::rank()>>({ get_start_index < 0 > ( Ranks ) ... }, { static_cast<std::int64_t>( view.extent ( Ranks ))... });
}

template <class Type, class Enabled = void>
struct CheckDuplicateEquality;

// Checks equality of individual element on floating points
template <class Type>
struct CheckDuplicateEquality<
Type, typename std::enable_if< std::is_floating_point < Type >::value, void >::type > {

  KOKKOS_INLINE_FUNCTION
  CheckDuplicateEquality() = default;

  KOKKOS_INLINE_FUNCTION
  CheckDuplicateEquality(const CheckDuplicateEquality& cde) = default;

  KOKKOS_INLINE_FUNCTION
  bool compare(Type a, Type b) const { return (abs(a - b) < 0.00000001); }
};

// Checks on non-floating points, user can create own checker for custom structs
template <class Type>
struct CheckDuplicateEquality<
Type, typename std::enable_if< !std::is_floating_point < Type >::value, void >::type > {

  KOKKOS_INLINE_FUNCTION
  CheckDuplicateEquality() = default;

  KOKKOS_INLINE_FUNCTION
  CheckDuplicateEquality(const CheckDuplicateEquality& cde) {}

  KOKKOS_INLINE_FUNCTION
  bool compare(Type a, Type b) const { return (a == b); }
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
  
  //This hack for a bool was replacable in the CUDA version and may be a c++ <17 artifact
  //Attempt to replace hack due to yucky error
  //Kokkos::View <bool*> success{"success", 1};
  mutable bool success = 1;

  static constexpr size_t rank = View::rank();
  // The rank of the view is known at compile-time, and there
  // is 1 subscriber per view. Therefore it is not templated on the 
  // instantiation of the original, but on the View itself 

  void clear() override
  {
    copy[0] = View ();
    copy[1] = View ();
  }

  bool execute() override
  { 
    //success(0) = true;
    success = true;

#ifdef KR_ENABLE_DMR
    if (duplicate_count < 1){
      Kokkos::abort("Aborted in CombineDuplicates, no duplicate created");
    }
#else
    if (duplicate_count < 2) {
      Kokkos::abort("Aborted in CombineDuplicates, duplicate_count < 2");
    }
#endif

    else {
      if constexpr(rank > 1){
	
     	//std::cout << "Rank is " << rank << " and this should be a multi-d p-for.\n";	      
        auto mdrange = make_md_range_policy( original, std::make_index_sequence< rank > {} ); 

        Kokkos::parallel_for(mdrange, *this);
	Kokkos::fence();


      }else{

	Kokkos::parallel_for("SubscriberCombiner1D", original.size(), *this);
        //Kokkos::fence();
      }
    }
    //return success(0);
    return success;
  }

  // Looping over duplicates to check for equality
  template<typename... Args> //template parameter pack
  KOKKOS_FUNCTION
  void operator ()(Args&&... its) const{ //function parameter pack

  //std::cout << "Test Print: Entered the combiner.\n\n";

#ifdef KR_ENABLE_DMR
    //Indicates dmr_failover_to_tmr tripped
    if(duplicate_count == 2 ){
      //Main Combiner begin, dmr failover has tripped into TMR
      for (int j = 0; j < 2; j ++) {
        if (check_equality.compare(copy[j](std::forward<Args>(its)...), original(std::forward<Args>(its)...))) {
          return;
        }
      }
      if (check_equality.compare(copy[0](std::forward<Args>(its)...), copy[1](std::forward<Args>(its)...))){
        original(std::forward<Args>(its)...) = static_cast<typename View::value_type>(copy[0](std::forward<Args>(its)...));  // just need 2 that are the same
	return;
      }
      //No match found, all three executions return different number
      //success(0) = false;
      success = false;
    }
    // DMR has not failed over, only 1 copy exists
    // Slight correction: 2 copies instantiated, 1 initialized
    else{
      //DMR combiner begin with no failover
      if (check_equality.compare(copy[0](std::forward<Args>(its)...), original(std::forward<Args>(its)...))){
	return;
      }
      //success(0) = false;
      success = false;
    }
#else
    //Main Combiner Begin
    for (int j = 0; j < 2; j ++) {
      if (check_equality.compare(copy[j](std::forward<Args>(its)...), original(std::forward<Args>(its)...))) {
        return;
      }
    }
    if (check_equality.compare(copy[0](std::forward<Args>(its)...), copy[1](std::forward<Args>(its)...))) {
      original(std::forward<Args>(its)...) = static_cast<typename View::value_type>(copy[0](std::forward<Args>(its)...));  // just need 2 that are the same
      return;
    }
    //No match found, all three executions return different number
    //success(0) = false;
    success = false;
#endif
  }

  void oneD_tmr_inject(){
    //Sequential, on original.size
    if ((original.size() != 1) && (ErrorInject::global_next_inject > original.size()))
    {
      ErrorInject::global_next_inject = ErrorInject::global_next_inject - original.size();
    }

    size_t next_inject = ErrorInject::global_next_inject;
#if 0
    double temp = global_error_settings->error_rate;
    std::cout << "Error::error_rate is " << temp << std::endl;

    std::cout << "ErrorInject::error_counter is " << ErrorInject::error_counter << std::endl;
    std::cout << "next_inject is " << next_inject << std::endl;
#endif
    for (int j=0; j<=2; j++){
      while (next_inject < original.size())
      {
        if (j==0){//Inject in the original if j is 0
          original(next_inject) = static_cast<typename View::value_type>(2 * original(next_inject) + 2 * ErrorInject::random_gen());//generate using ()
          ErrorInject::error_counter++;
        }
        else{//Else inject in one of the other two copies, copy[0] or copy[1]
          copy[j-1](next_inject) = static_cast<typename View::value_type>(2 * copy[j-1](next_inject) + 2 * ErrorInject::random_gen());
          ErrorInject::error_counter++;
        }
        next_inject = global_error_settings->geometric(ErrorInject::random_gen)+next_inject+1;

      }
      if(original.size() != 1){
      next_inject = (next_inject) - (original.size());
      }
    }
  }

  void TwoDimTMRInject(){
//#if 0    
    //This error injection works with 2D views, subtracts total extent from global_next_inject
    size_t total_extent = original.extent(0) * original.extent(1);
    if (total_extent !=1 && (ErrorInject::global_next_inject > total_extent))
    {
      ErrorInject::global_next_inject = ErrorInject::global_next_inject - total_extent;
    }

    size_t next_inject = ErrorInject::global_next_inject;
#if 0    
    double temp = global_error_settings->error_rate;
    std::cout << "Error::error_rate is " << temp << std::endl;
    
    std::cout << "ErrorInject::error_counter is " << ErrorInject::error_counter << std::endl;
    std::cout << "original.extent(0) is " << original.extent(0) << std::endl;
    std::cout << "original.extent(1) is " << original.extent(1) << std::endl;
    std::cout << "total_extent is " << total_extent << std::endl;
    std::cout << "next_inject is " << next_inject << std::endl;
#endif
#if 0
    //Completely closed off print loop. DELETE!
    for (int j=0; j<=2; j++){
      while (next_inject < total_extent)
      {
	std::cout << "The value at next_inject translates to array(" << floor(next_inject/original.extent(0)) << "," 
		  << next_inject - (original.extent(0) * floor(next_inject/original.extent(0))) << ") = "
		  << static_cast<typename View::value_type>(original((int)floor(next_inject/original.extent(0)),next_inject - (original.extent(0) * (int)floor(next_inject/original.extent(0)))))
		  << "." << std::endl; 

	ErrorInject::error_counter++;
	next_inject = global_error_settings->geometric(ErrorInject::random_gen)+next_inject+1;
	std::cout << "next_inject is " << next_inject << std::endl;
      }
      if(total_extent != 1){
        next_inject = next_inject - total_extent;
      }
    }
#endif


//#if 0
    for (int j = 0; j<=2; j++){
      while (next_inject < total_extent)
      {
#if 0
		  std::cout << "The value at next_inject translates to array(" << floor(next_inject/original.extent(0)) << "," 
                  << next_inject - (original.extent(0) * floor(next_inject/original.extent(0))) << ") = "
                  << static_cast<typename View::value_type>(original((int)floor(next_inject/original.extent(0)),next_inject - (original.extent(0) * (int)floor(next_inject/original.extent(0)))))
                  << "." << std::endl; 
#endif
        next_inject = global_error_settings->geometric(ErrorInject::random_gen)+next_inject+1;
        //std::cout << "next_inject is " << next_inject << std::endl;
      
	      
        if (j==0){//Inject in the original if j is 0
          original((int)floor(next_inject/original.extent(0)), next_inject % original.extent(0)) 
		  = static_cast<typename View::value_type>( 
		                2 * original((int)floor(next_inject/original.extent(0)), next_inject % original.extent(0)) 
				+ 2 * ErrorInject::random_gen());//generate using ()
          ErrorInject::error_counter++;
        }
        else{//Else inject in one of the other two copies, copy[0] or copy[1]
          copy[j-1]((int)floor(next_inject/original.extent(0)), next_inject % original.extent(0)) 
		  = static_cast<typename View::value_type>( 
		            2 * copy[j-1]((int)floor(next_inject/original.extent(0)), next_inject % original.extent(0)) 
			    + 2 * ErrorInject::random_gen());
          ErrorInject::error_counter++;
        }
        next_inject = global_error_settings->geometric(ErrorInject::random_gen)+next_inject+1;
        //std::cout << "next_inject is " << next_inject << std::endl;
      }
      if(total_extent != 1){
        next_inject = next_inject - total_extent;
      }
    }
//#endif
  }

  void inject_error() override
  {
	  
    //std::cout << "We got into the error injector and rank is: " << rank << "\n";
    if constexpr(rank == 2){
#ifdef KR_ENABLE_DMR
      //Implies dmr_failover_to_tmr
      if(duplicate_count == 2) {
        //goto main tmr inject
	TwoDimTMRInject();
      }
      else{//Actual DMR error injection with only 1 copy
      }//End DMR error injection
#else
      //Working not perfect
      //std::cout << "We got into the 2d error injector section\n";
      TwoDimTMRInject();
#endif 
    }else{

#ifdef KR_ENABLE_DMR
      //Implies dmr_failover_to_tmr has tripped    
      if(duplicate_count == 2 ){
	//goto main_tmr_inject;
	oneD_tmr_inject();
      }
      else{//The actual dmr error inject with only 1 copy
        if ((original.size() != 1) && (ErrorInject::global_next_inject > original.size()))
        {
          ErrorInject::global_next_inject = ErrorInject::global_next_inject - original.size();
        }

        size_t next_inject = ErrorInject::global_next_inject;
	for (int j = 0; j<2; j++){
          while (next_inject < original.size())
          {
            if (j==0){//Insert error into original
		    //std::cout <<"Injecting an error at " <<next_inject<< " in the original"<< std::endl;	    
              original(next_inject) = static_cast<typename View::value_type>(2 * original(next_inject) + 2 * ErrorInject::random_gen());//generate using ()
              ErrorInject::error_counter++;
            }
            else{//Insert error into only copy
		    //std::cout <<"Injecting an error at " <<next_inject<< " in copy [0]"<< std::endl;
              copy[0](next_inject) = static_cast<typename View::value_type>(2 * copy[0](next_inject) + 2 * ErrorInject::random_gen());
              ErrorInject::error_counter++;
            }
            next_inject = global_error_settings->geometric(ErrorInject::random_gen)+next_inject+1;
          }
          if(original.size() != 1){
          next_inject = (next_inject) - (original.size());
          }
        }
      }

#else
	oneD_tmr_inject();
#endif
    }//end rank == 1
  }// end error inject

};// end Combiner

} // namespace KokkosResilience

/*----------------------------------------------------------------------------
 ************ SUBSCRIBER TO DUPLICATE VIEWS, COPY DATA ***********************
 ----------------------------------------------------------------------------*/

namespace KokkosResilience {

struct ResilientDuplicatesSubscriber {

  // Gating for using subscriber only inside resilient parallel loops
  static bool in_resilient_parallel_loop;

#ifdef KR_ENABLE_DMR
  static bool dmr_failover_to_tmr;

#endif

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

#if 0    
#ifdef KR_ENABLE_DMR

     // DMR variant initialization of copies
     // only 1 initially, 2 on fail
     if (dmr_failover_to_tmr && res.copy[1].data()==nullptr){
      // Create second copy
        set_duplicate_view(res.copy[1], original, 1);
      }
#endif
#endif

    if (inserted || extents_resized) {
      res.original = original;

#ifdef KR_ENABLE_DMR
      if (dmr_failover_to_tmr){
        // Create second copy
        set_duplicate_view(res.copy[1], original, 1);
      }else{

	      //TODO: for dmr want this to be in main inserted if
	      //and uncomment if 0
      // only 1 initially, 2 on fail
      // Create first copy for DMR
      set_duplicate_view(res.copy[0], original, 0);
      }
#else

      // Reinitialize self to be like other (same dimensions, etc)
      for (int i = 0; i < 2; ++i) {
          set_duplicate_view(res.copy[i], original, i);
      }

#endif

    }
    return &res;
  }
  // Function which initializes the dimensions of the duplicating view
  template<typename View>
  KOKKOS_INLINE_FUNCTION
  static void set_duplicate_view(View &duplicate, const View &original, int duplicate_count) {
    std::stringstream label_ss;
    label_ss << original.label() << duplicate_count;
    duplicate = View(label_ss.str(), original.layout());
  }

/*
  // A template argument V (view), a template itself, having at least one parameter
  // the first one (T), to determine between the const/non-const copy constructor in overload
  // Class because C++14
  template<template<typename, typename ...> class V, typename T, typename... Args>
  static void copy_constructed( V < const T *, Args...> &self, const V < const T *, Args...> &other) {
    // If View is constant do nothing, not triggering the rest of the subscriber.
  }

  template< template< typename, typename ...> class V, typename T, typename... Args>
  static void copy_constructed( V < T *, Args... > &self, const V < T *, Args... > &other)
  {
*/

  template<typename View>
  static void copy_constructed( View &self, const View &other)
  {
    //if constexpr ( !std::is_const_v< typename V::value_type > )
    if constexpr( std::is_same_v< typename View::non_const_data_type, typename View::data_type > )
    {
      // If view is non-constant and in the parallel loop, cascade the rest of the subscriber
      if (in_resilient_parallel_loop) {
        // This won't be triggered if the entry already exists
        auto *combiner = get_duplicate_for(other);
        auto res = duplicates_map.emplace(std::piecewise_construct,
                                          std::forward_as_tuple(other.data()),
                                          std::forward_as_tuple(combiner));
        //auto &c = dynamic_cast< CombineDuplicates< V<T*, Args...> > & > (*res.first->    second);
    	auto &c = dynamic_cast< CombineDuplicates< View > & > (*res.first->second);

        // The first copy constructor in a parallel_for for the given view
        if (res.second) {
          c.duplicate_count = 0;
        }

        self = c.copy[c.duplicate_count++];
        // Copy all data, every time
        Kokkos::deep_copy(self, other);
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

KOKKOS_INLINE_FUNCTION
void print_total_error_time() {
  //std::lock_guard<std::mutex> lock(ETimer::global_time_mutex);

  ETimer::global_time_mutex.lock();

  //actual time (in seconds) of duration d = d.count() * D::period::num / D::period::den
  //D::period::num = 1
  //D::period::den = 1000000000
  //Selected because tick rate of stable_clock
  //auto time = ETimer::total_error_time.count() / 1000000000;
  //const auto time = std::chrono::duration_cast<std::chrono::seconds>(ETimer::total_error_time);
  std::cout << "The value of ETimer::total_error_time.count() is " << ETimer::total_error_time.count() << " nanoseconds." << std::endl;
  std::cout << "The total number of errors inserted is " << ErrorInject::error_counter << " errors." << std::endl;
  ETimer::global_time_mutex.unlock();

}

KOKKOS_INLINE_FUNCTION
void clear_duplicates_map() {

  KokkosResilience::ResilientDuplicatesSubscriber::duplicates_map.clear();
}

KOKKOS_INLINE_FUNCTION
void clear_duplicates_cache() {
  
  // Go over the Subscriber map, deallocate copies contained in the CombinerBase elements
  for (auto&& combiner : KokkosResilience::ResilientDuplicatesSubscriber::duplicates_cache) {
    combiner.second->clear();
  }
  KokkosResilience::ResilientDuplicatesSubscriber::duplicates_cache.clear();
}

} //namespace KokkosResilience

#endif //defined(KOKKOS_ENABLE_OPENMP)
#endif
