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
#include "OpenMPResSubscriber.cpp"

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
    Type, typename std::enable_if< std::is_floating_point < Type >::value, void >::type > {

    KOKKOS_INLINE_FUNCTION
  CheckDuplicateEquality() {}

  KOKKOS_INLINE_FUNCTION
  CheckDuplicateEquality(const CheckDuplicateEquality& cde) {}

  KOKKOS_INLINE_FUNCTION
  bool compare(Type a, Type b) const { return (abs(a - b) < 0.00000001); }
};

template <class Type>
struct CheckDuplicateEquality<
    Type, typename std::enable_if< !std::is_floating_point < Type >::value, void >::type > {

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

  int duplicate_count = 0;
  View original;
  View copy[3];
  bool success = false;

  bool execute() override
  {
    if (duplicate_count < 3){
      Kokkos::abort("Aborted in CombineDuplicates, duplicate_count < 3");
    }
    else {
      success = false;
      //TODO: WIL MULTIDIMENSIONAL VIEW AFFECT? TEST (MIGHT NEED EXECUTION POLICY)
      //Kokkos::parallel_for(original.size(), *this);
      Kokkos::parallel_for(original.size(), KOKKOS_LAMBDA(int i){
        *this;
      });
    }
    return success;

  }

  //KOKKOS_FUNCTION
  KOKKOS_INLINE_FUNCTION
  void operator ()(int i) {

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
          success = true;
          return;
        }
        k = k < 2 ? k + 1 : 0;
      }
    }
    //No match found, all three executions return different number
    printf("no match found: %i\n", i);
    // TODO: MOVE ABORT HERE FROM MAIN COMBINE CALL (TESTING)
    // TODO: DISCUSS WITH NIC, NOT ABORT, JUST BREAK INTO MAIN P_FOR
    success = false;
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

  //TODO: Write map clearing function

  //Is the view_like function which initialize the dimensions of the duplicating view
  template <typename View>
  KOKKOS_INLINE_FUNCTION
  static void ViewMatching(View &self, const View &other) {
    self = View();
    // According to view API wiki...
    self.layout() = other.layout();
  }

  template <typename View>
  static void copy_constructed(View &self, const View &other) {
    if (in_resilient_parallel_loop) {

      //This won't be triggered if the entry already exists
      auto res = duplicates_map.emplace(std::piecewise_construct,
                                        std::forward_as_tuple(other.data()),
                                        std::forward_as_tuple(std::make_unique< CombineDuplicates< View, typename View::data_type > >()) );
      auto &c = dynamic_cast< CombineDuplicates < View, typename View::data_type >& > (*res.first->second);

      if (res.second){c.original = other;}

      in_resilient_parallel_loop = false;
      // Reinitialize self to be like other (same dimensions, etc)
      ViewMatching(self, other);

      // Copy all data
      Kokkos::deep_copy(self, other);

      // Reference counting can be turned off here for performance reasons
      c.copy[c.duplicate_count++] = self;
    }
  }
};

KOKKOS_INLINE_FUNCTION
void clear_duplicates_map() {
  for (auto &&entry : KokkosResilience::ResilientDuplicatesSubscriber::duplicates_map) {
    // Just delete entry, delete pointer as well? Deleting by key fine, or need to delete combiner struct?
    KokkosResilience::ResilientDuplicatesSubscriber::duplicates_map.erase(entry.first);
  }
}

} //namespace KokkosResilience

#endif //defined(KOKKOS_ENABLE_OPENMP) //&& defined (KR_ENABLE_ACTIVE_EXECUTION_SPACE)
#endif