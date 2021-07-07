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
#if defined(KOKKOS_ENABLE_OPENMP) //&& defined (KR_ENABLE_ACTIVE_EXECUTION_SPACE)

#include <omp.h>
#include <iostream>
#include <OpenMP/Kokkos_OpenMP_Exec.hpp>
#include <impl/Kokkos_FunctorAdapter.hpp>

#include <KokkosExp_MDRangePolicy.hpp>

#include <Kokkos_Parallel.hpp>
#include <OpenMP/Kokkos_OpenMP_Parallel.hpp>
#include <Kokkos_Core_fwd.hpp>
#include <cmath>
#include <map>
#include <typeinfo>
#include <unordered_map>

/*--------------------------------------------------------------------------
 *************** TEST SUBSCRIBER, DELETE LATER *****************************
 --------------------------------------------------------------------------*/

struct TestIISubscriber;

// Generate usable error message
static_assert( Kokkos::Experimental::is_hooks_policy< Kokkos::Experimental::SubscribableViewHooks< TestIISubscriber > >::value, "Must be a hooks policy" );

struct TestIISubscriber
{
  static Kokkos::View< double **, Kokkos::Experimental::SubscribableViewHooks< TestIISubscriber > >* self_ptr;
  static const Kokkos::View< double **, Kokkos::Experimental::SubscribableViewHooks< TestIISubscriber > >* other_ptr;

  template< typename View >
  static void copy_constructed( View &self, const View &other )
  {
    self_ptr= &self;
    other_ptr = &other;

  }
};

/*--------------------------------------------------------------------------
 ******************** ERROR MESSAGE GENERATION *****************************
 --------------------------------------------------------------------------*/

// UH OH NAMESPACE AGREEMENT!!!!!!!!!!!

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

template <class Type>
struct CheckDuplicateEquality<
    Type, typename std::enable_if<std::is_same<Type, float>::value ||
                                  std::is_same<Type, double>::value,
        void>::type> {
  KOKKOS_INLINE_FUNCTION
  CheckDuplicateEquality() {}

  KOKKOS_INLINE_FUNCTION
  CheckDuplicateEquality(const CheckDuplicateEquality& cde) {}

  KOKKOS_INLINE_FUNCTION
  bool compare(Type a, Type b) const { return (abs(a - b) < 0.00000001); }
};

template <class Type>
struct CheckDuplicateEquality<
    Type, typename std::enable_if<!std::is_same<Type, float>::value &&
                                  !std::is_same<Type, double>::value,
        void>::type> {
  KOKKOS_INLINE_FUNCTION
  CheckDuplicateEquality() {}

  KOKKOS_INLINE_FUNCTION
  CheckDuplicateEquality(const CheckDuplicateEquality& cde) {}

  KOKKOS_INLINE_FUNCTION
  bool compare(Type a, Type b) const { return (a == b); }
};

} // namespace KokkosResilience

namespace KokkosResilience{

struct CombineDuplicatesBase
{
  // Need virtual bool in order to return success
  // TODO: clean comment
  virtual bool execute() = 0;
};

template< typename View , class DuplicateType >
struct CombineDuplicates: public CombineDuplicatesBase
{
  using EqualityType = CheckDuplicateEquality<DuplicateType>;
  EqualityType check_equality;

  int duplicate_count;
  View original;
  View copy[3];
  size_t original_len;

  // TODO: HOW IS original_len ACQUIRING VALUE original.size()
  // HOW IS duplicate_count acquiring 3? It's not! Need to assign it in subscriber, attach to...?

  bool execute() override
  {
    if (duplicate_count < 3){
      Kokkos::abort("Aborted in CombineDuplicates, duplicate_count < 3");
    }
    else {
      // TODO: WILL BE PROBLEM, RETURNS BOOL. SAME ISSUE AS BEFORE, REDUCE ON SUCCESS?
      Kokkos::parallel_for(original_len, *this);
    }

  }

  KOKKOS_INLINE_FUNCTION
  bool operator ()(const int i) const {
    //KOKKOS_FUNCTION
    //bool operator () (int i)
  // need bool for success, syntax??
  // TODO: delete comment
  // Move specialization from exec space to this file?
  // local.exec function with trigger ->> trigger fix later

    //TODO: IS THE = DOING WHAT I THINK IN VIEWS???

    //printf("Majority vote, index i: %d\n", i);
    for (int j = 0; j < 3; j++) {
      //printf("Outer iteration: %d - %d \n", i, j);
      original(i) = copy[j](i);
      //printf("first entry: %d, %d\n", j, orig_view[i]);
      int k = j < 2 ? j + 1 : 0;
      for (int r = 0; r < 2; r++) {
        //printf("iterate inner %d, %d, %d \n", i, j, k);
        if (check_equality.compare(copy[k](i),
                       original(i)))  // just need 2 that are the same
        {
          printf("match found: %d - %d\n", k, j);
          return 1;
        }
        k = k < 2 ? k + 1 : 0;
      }
    }
    //No match found, all three executions return different number
    printf("no match found: %i\n", i);
    // TODO: MOVE ABORT HERE FROM MAIN COMBINE CALL (TESTING)
    // TODO: DISCUSS WITH NIC, NOT ABORT, JUST BREAK INTO MAIN P_FOR
    return 0;
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

  // TODO: ASK NIC, HOW IS THIS MAP CLEARING? Need 3 duplicates attached.

  // Need to count how many duplicates attached to a particular data() pointer
  //int duplicate_count;

  // Attach it some other way?
  //void* duplicate_list[3];

  //Is the view_like function which initialize the dimensions of the duplicating view
  KOKKOS_INLINE_FUNCTION
  ViewMatching(View &self, const View &other) {
    // According to view API wiki...
    self.layout() = other.layout();
  }

  template <typename View>
  static void copy_constructed(View &self, const View &other) {
    if (in_resilient_parallel_loop) {
      auto &c = duplicates_map[other.data()];
      c.size  = other.extent();
      // will do reference counting, etc... shares allocation record
      c.original = other;

      in_resilient_parallel_loop = false;
      // Reinitialize self to be like other (same dimensions, etc)
      self = ViewMatching(other);

      // Copy all data
      Kokkos::deep_copy(self, other);

      // Reference counting can be turned off here for performance reasons
      c.copy[c.count++] = self;
    }
  }
};

} //namespace KokkosResilience

#endif //defined(KOKKOS_ENABLE_OPENMP) //&& defined (KR_ENABLE_ACTIVE_EXECUTION_SPACE)
#endif