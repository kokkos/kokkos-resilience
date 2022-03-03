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

#ifndef INC_RESILIENCE_OPENMP_OPENMPRESSUBSCRIBER_HPP
#define INC_RESILIENCE_OPENMP_OPENMPRESSUBSCRIBER_HPP

#include <Kokkos_Macros.hpp>
#if defined(KOKKOS_ENABLE_OPENMP)

#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_CopyViews.hpp>
#include <omp.h>
#include <iostream>
#include <Kokkos_OpenMP.hpp>
#include <KokkosExp_MDRangePolicy.hpp>
#include <impl/Kokkos_FunctorAdapter.hpp>
#include <Kokkos_Parallel.hpp>
#include <cmath>
#include <map>
#include <typeinfo>
#include <unordered_map>
#include <sstream>

/*--------------------------------------------------------------------------
 ******************** ERROR MESSAGE GENERATION *****************************
 --------------------------------------------------------------------------*/

namespace KokkosResilience {

struct ResilientDuplicatesSubscriber;

// Generate usable error message
static_assert(Kokkos::Experimental::is_hooks_policy<
                  Kokkos::Experimental::SubscribableViewHooks<
                      ResilientDuplicatesSubscriber> >::value,
              "Must be a hooks policy");
}

/*----------------------------------------------------------------------------
 ******** STRUCT TO CHECK CORRECTNESS OF INDIVIDUAL ELEMENTS OF VIEWS ********
 ----------------------------------------------------------------------------*/

namespace KokkosResilience {

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
  virtual bool execute() = 0;
  virtual void print() = 0;
};

template< typename View >
struct CombineDuplicates: public CombineDuplicatesBase
{
  using EqualityType = CheckDuplicateEquality<typename View::value_type>;
  EqualityType check_equality;

  int duplicate_count = 0;
  View original;
  View copy[3];
  Kokkos::View <bool*> success{"success", 1};

  bool execute() override
  {
    success(0) = false;

    if (duplicate_count < 3){
      Kokkos::abort("Aborted in CombineDuplicates, duplicate_count < 3");
    }
    else {

      Kokkos::parallel_for(original.size(), *this);
      Kokkos::fence();
    }
    return success(0);
  }

  void print () override {
    std::cout << "This is the original data pointer " << original.data() << std::endl;
    std::cout << "This is copy[0] data pointer " << copy[0].data() << std::endl;
    std::cout << "This is copy[1]  data pointer " << copy[1].data() << std::endl;
    std::cout << "This is copy[2]  data pointer " << copy[2].data() << std::endl;
/*
    for (int i=0; i<original.size();i++){
      std::cout << "This is the original at index " << i << " with value" << original(i) << std::endl;
      std::cout << "This is copy[0] at index " << i << " with value" << copy[0](i) << std::endl;
      std::cout << "This is copy[1] at index " << i << " with value" << copy[1](i) << std::endl;
      std::cout << "This is copy[2] at index " << i << " with value" << copy[2](i) << std::endl;

    }*/
  }

  // Looping over duplicates to check for equality
  template <typename ... T>
  KOKKOS_FUNCTION
  void operator ()(T ... is) const {

    for (int j = 0; j < 3; j++) {
        //printf("Original value before compare at index %d is %lf\n", i, original(i));
        //printf("Copy[%d] value before compare at index %d is %lf\n", j, i, copy[j](i));
        //printf("Outer iteration: %d - %d \n", i, j);
      original(is...) = copy[j](is...);

      for (int r = 0; r < 2; r++) {
        int k = (j+r+1)%3;

        if (check_equality.compare(copy[k](is...),
                       original(is...)))  // just need 2 that are the same
        {
          //printf("match found: %d - %d\n", k, j);
          //printf("Original value after compare at index %d is %lf\n", i, original(i));
          //printf("Copy[%d] value after compare at index %d is %lf\n", k, i, copy[k](i));
          Kokkos::atomic_assign(&success(0), true);
          return;
        }
      }
    }

    //No match found, all three executions return different number
    //printf("no match found: %i\n", i);
    Kokkos::atomic_assign(&success(0), false);
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

  // Creating map for duplicates
  using key_type = void *;  // key_type should be data() pointer
  static std::unordered_map<key_type, std::unique_ptr<CombineDuplicatesBase> > duplicates_map;

  // Function which initializes the dimensions of the duplicating view
  template <typename View>
  KOKKOS_INLINE_FUNCTION
  static void ViewMatching(View &self, const View &other, int duplicate_count) {

    std::stringstream label_ss;
    label_ss << other.label() << duplicate_count;
    self = View(label_ss.str(), other.layout());

  }

  template <typename View>
  static void copy_constructed(View &self, const View &other) {
    if (in_resilient_parallel_loop) {

      //This won't be triggered if the entry already exists
      auto res = duplicates_map.emplace(std::piecewise_construct,
                                        std::forward_as_tuple(other.data()),
                                        std::forward_as_tuple(std::make_unique< CombineDuplicates< View > >()) );
      auto &c = dynamic_cast< CombineDuplicates < View >& > (*res.first->second);

      if (res.second){c.original = other;}

      // Reinitialize self to be like other (same dimensions, etc)
      ViewMatching(self, other, c.duplicate_count);

      // Copy all data
      Kokkos::deep_copy(self, other);

      // Reference counting can be turned off here for performance reasons
      c.copy[c.duplicate_count++] = self;
    }
  }
};

KOKKOS_INLINE_FUNCTION
void clear_duplicates_map() {
  KokkosResilience::ResilientDuplicatesSubscriber::duplicates_map.clear();
}

KOKKOS_INLINE_FUNCTION
void print_duplicates_map(){
  for(auto &&entry : KokkosResilience::ResilientDuplicatesSubscriber::duplicates_map) {
    entry.second->print();
  }
}

} //namespace KokkosResilience

#endif //defined(KOKKOS_ENABLE_OPENMP)
#endif
