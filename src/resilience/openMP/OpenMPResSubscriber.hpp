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

explicit Error(uint64_t seed, double threshold) : local_seed(seed), local_random_pool(seed), error_rate(threshold) {}
uint64_t local_seed;
Kokkos::Random_XorShift64_Pool<> local_random_pool;
double error_rate;

/*
//Error timer settings
thread_local double elapsed_seconds;
double total_error_time;

//Mutex for total_error_time accumulation
std::mutex global_time_mutex;
*/

};

inline std::optional<Error> global_error_settings;

/*
// Struct to gate error timing
struct ETimer{
       
explicit ETimer	(int subscriber_max_pool) : subcriber_max_threads (subcriber_max_pool){}
int subscriber_max_threads;
//global thread_local durations for error accumulation in combiner accross kernel
thread_local std::chrono::duration<double> elapsed_seconds;
//global duration for error accumulation accross iterations of kernels
auto std::chrono::duration<double> total_error_time;

};	

inline std::optional<ETimer> global_timer_variables;
*/

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
  virtual bool execute() = 0;
  //virtual void insert_error();
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

//#if 0
  //Setup for error insertion
  //Scope is one CombineDuplicates struct (one view), re-initializes per view?
  //Seeds see paper?
    //uint64_t seed = 12345;
    
    //Global variable in heatdis:
    //templated on <class DeviceType>, should auto-get?
    //Kokkos::Random_XorShift64_Pool<> random_pool{seed};
//#endif

  //This hack for a bool was replacable in the CUDA version and may be a c++ <17 artifact
  Kokkos::View <bool*> success{"success", 1};

  static constexpr size_t rank = View::rank();
  // The rank of the view is known at compile-time, and there
  // is 1 subscriber per view. Therefore it is not templated on the 
  // instantiation of the original, but on the View itself 

  bool execute() override
  {
    success(0) = true;

    

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
         
//Kokkos::Profiling::pushRegion("SubCombinerMultiD");

        auto mdrange = make_md_range_policy( original, std::make_index_sequence< rank > {} ); 

        Kokkos::parallel_for(mdrange, *this);
	Kokkos::fence();

//Kokkos::Profiling::popRegion();

      }else{

//Kokkos::Profiling::pushRegion("SubCombiner1D");

        Kokkos::parallel_for("SubscriberCombiner1D", original.size(), *this);
        Kokkos::fence();

	/*
        //Each thread will add their own thread_local global version of elapsed_seconds
        #pragma omp parallel
	{
          std::lock_guard<std::mutex> lock(global_error_settings->global_time_mutex);
          global_error_settings->total_error_time{ global_error_settings->total_error_time + global_error_settings->elapsed_seconds };

        }*/

//Kokkos::Profiling::popRegion();

      }
    }
    return success(0);
  }

  // Looping over duplicates to check for equality
  template<typename... Args> //template parameter pack
  KOKKOS_FUNCTION
  void operator ()(Args&&... its) const { //function parameter pack

#ifdef KR_ENABLE_DMR
//ERROR INSERTION 0 GATE HERE    
   

    //Indicates dmr_failover_to_tmr tripped
    if(duplicate_count == 2 ){

//Error Insertion Code
#if 0
      auto generator = random_pool.get_state();

      double x = generator.drand(0., 1.);
      double y = generator.drand(0., 1.);
      double z = generator.drand(0., 1.);

      random_pool.free_state(generator);

      if(x<0.00001) 
        original(std::forward<Args>(its)...) = 2 * static_cast<typename View::value_type>(original(std::forward<Args>(its)...)) + 2 * x;
      if(y<0.00001)
        copy[0](std::forward<Args>(its)...) = 2 * static_cast<typename View::value_type>(copy[0](std::forward<Args>(its)...) + 2 * y;
      if(z<0.00001)
        copy[1](std::forward<Args>(its)...) = 2 * static_cast<typename View::value_type>(copy[1](std::forward<Args>(its)...) + 2 * z;
#endif
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
      success(0) = false;
    }
    // DMR has not failed over, only 1 copy exists
    // Slight correction: 2 copies instantiated, 1 initialized
    else{
//Error Insertion Code
#if 0
      auto generator = random_pool.get_state();

      double x = generator.drand(0., 1.);
      double y = generator.drand(0., 1.);

      random_pool.free_state(generator);

      if(x<0.00001) 
        original(std::forward<Args>(its)...) = 2 * static_cast<typename View::value_type>(original(std::forward<Args>(its)...)) + 2 * x;
      if(y<0.00001)
        copy[0](std::forward<Args>(its)...) = 2 * static_cast<typename View::value_type>(copy[0](std::forward<Args>(its)...)) + 2 * y;
      
#endif
//DMR Combiner Begin with no failover
      if (check_equality.compare(copy[0](std::forward<Args>(its)...), original(std::forward<Args>(its)...))){
        return;
      }
      success(0) = false;
    }

#else
//Error Insertion Code
//#if 0
    if(global_error_settings){
    
    //Steady-clock should be thread-safe
   // const auto start{std::chrono::steady_clock::now()};    

//Kokkos::Profiling::pushRegion("ErrorGeneration");	    

    auto generator = global_error_settings->local_random_pool.get_state();
 
    double x = generator.drand(0., 1.);
    double y = generator.drand(0., 1.);
    double z = generator.drand(0., 1.);
  
    global_error_settings->local_random_pool.free_state(generator);
   
    if(x < global_error_settings->error_rate){
      // printf("An error was inserted in the original\n");
      original(std::forward<Args>(its)...) = 2 * static_cast<typename View::value_type>(original(std::forward<Args>(its)...)) + 2 * x;
      }
    if(y < global_error_settings->error_rate){
      // printf(An error was inserted in the first copy\n);
      copy[0](std::forward<Args>(its)...) = 2 * static_cast<typename View::value_type>(copy[0](std::forward<Args>(its)...)) + 2 * y;
      }
    if(z < global_error_settings->error_rate){
      // printf("An error was inserted in the second copy\n");
      copy[1](std::forward<Args>(its)...) = 2 * static_cast<typename View::value_type>(copy[1](std::forward<Args>(its)...)) + 2 * z;
      }

    //const auto stop{std::chrono::steady_clock::now()};

    //global_error_settings->elapsed_seconds{ elapsed_seconds + (stop - start) };
//Kokkos::Profiling::popRegion();

    }
//endif

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
    success(0) = false;
#endif
  }
};

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
    if (inserted) {
      res.original = original;

#ifdef KR_ENABLE_DMR

      // DMR variant initialization of copies
      // only 1 initially, 2 on fail
      if (dmr_failover_to_tmr){
        // Create second copy
        set_duplicate_view(res.copy[1], original, 1);
      }
      else{
        // Create first copy
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
    // If view is non-constant and in the parallel loop, cascade the rest of the subscriber
    if (in_resilient_parallel_loop) {

      // This won't be triggered if the entry already exists
      auto *combiner = get_duplicate_for(other);
      auto res = duplicates_map.emplace(std::piecewise_construct,
                                        std::forward_as_tuple(other.data()),
                                        std::forward_as_tuple(combiner));
      auto &c = dynamic_cast< CombineDuplicates< V<T*, Args...> > & > (*res.first->second);

      // The first copy constructor in a parallel_for for the given view
      if (res.second) {
        c.duplicate_count = 0;
      }

      self = c.copy[c.duplicate_count++];
      // Copy all data, every time
      Kokkos::deep_copy(self, other);
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
void clear_duplicates_map() {
  KokkosResilience::ResilientDuplicatesSubscriber::duplicates_map.clear();
}

KOKKOS_INLINE_FUNCTION
void clear_duplicates_cache() {
  KokkosResilience::ResilientDuplicatesSubscriber::duplicates_cache.clear();
}

} //namespace KokkosResilience

#endif //defined(KOKKOS_ENABLE_OPENMP)
#endif
