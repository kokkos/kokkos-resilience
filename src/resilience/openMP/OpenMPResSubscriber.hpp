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

/*--------------------------------------------------------------------------
 *************** TEST SUBSCRIBER, DELETE **********************************
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

/*----------------------------------------------------------------------------
 ************ SUBSCRIBER TO DUPLICATE VIEWS, COPY DATA ***********************
 ----------------------------------------------------------------------------*/

/*
namespace KokkosResilience {

struct ResilientDuplicatesSubscriber {
  static bool in_resilient_parallel_loop;

  using key_type = void *;  // could be view name
  static std::unordered_map<key_type, std::unique_ptr<CombineDuplicatesBase> > map;

  template <typename View>
  static void copy_constructed(View &self, const View &other) {
    if (in_resilient_parallel_loop) {
      auto &c = map[other.data()];
      c.size  = other.extens();
      // will do reference counting, etc... shares allocation record
      c.original = other;

      in_resilient_parallel_loop = false;
      // Reinitialize self to be like other (same dimensions, etc)
      self = view_like(other);

      // Copy all data
      Kokkos::deep_copy(self, other);

      // Reference counting can be turned off here for performance reseasons
      c.copy[c.count++] = self;
    }
  }
};

} //namespace KokkosResilience

/*----------------------------------------------------------------------------
 ************ COMBINER TO RECOMBINE VIEWS, CHECK CORRECTNESS *****************
 ----------------------------------------------------------------------------*/
/*
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

struct CombineDuplicateBase
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

  View original;
  View copy[3];
  //size_t = m_len;

  KOKKOS_INLINE_FUNCTION
  // void operator ()(const int i) const {
  // need bool for success
  // TODO: delete comment
  // Move caller from exec space to this file?
  // local.exec function with trigger ->> trigger fix later

  bool exec(const int i) const {

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
    return 0;
  }

};

} // namespace KokkosResilience

//

/*----------------------------------------------------------------------------
 ************ COMBINER CALL MOVED FROM EXEC SPACE ****************************
 ----------------------------------------------------------------------------*/
/*
namespace KokkosResilience{

template <class Type>
class SpecDuplicateTracker<Type, Kokkos::OpenMP> : public DuplicateTracker {
 public:
  typedef typename std::remove_reference<Type>::type nr_type;
  typedef typename std::remove_pointer<nr_type>::type np_type;
  typedef typename std::remove_extent<np_type>::type ne_type;
  typedef typename std::remove_const<ne_type>::type rd_type;
  typedef CombineFunctor<rd_type, Kokkos::OpenMP> comb_type;

  comb_type m_cf;

  inline SpecDuplicateTracker() : DuplicateTracker(), m_cf() {}

  inline SpecDuplicateTracker(const SpecDuplicateTracker& rhs)
      : DuplicateTracker(rhs), m_cf(rhs.m_cf) {}

  virtual bool combine_dups();
  virtual void set_func_ptr();
};

template <class Type>
void SpecDuplicateTracker<Type, Kokkos::OpenMP>::set_func_ptr() {}

template <class Type>
bool SpecDuplicateTracker<Type, Kokkos::OpenMP>::combine_dups() {

  //bool success;
  bool trigger = 1;

  if (dup_cnt != 3) {
    printf("must have 3 duplicates !!!\n");
    fflush(stdout);
    return 0;
  }
  int N = data_len / sizeof(rd_type);
  m_cf.load_ptrs( static_cast<rd_type*>(original_data)
      , static_cast<rd_type*>(dup_list[0])
      , static_cast<rd_type*>(dup_list[1])
      , static_cast<rd_type*>(dup_list[2]), N );

  comb_type local_cf(m_cf);
  printf("Combine duplicates has size N=%d\n\n\n",N);
  fflush(stdout);


  Kokkos::parallel_for( N, KOKKOS_LAMBDA(int i) {
    local_cf.exec(i);
  });
*/
/*
  //currently counting if there are data-length number of successes
  //that is, everything successful
  //TODO: Change to only one unsuccessful

  int result = 0;
  Kokkos::parallel_reduce( "combine_dups", N, KOKKOS_LAMBDA( int i, int &success ) {
    success += local_cf.exec(i);
  }, result );

  printf("RESULT = %d after combining\n", result);
  fflush(stdout);

  printf("trigger = %d after combining but not setting (should be 1) \n", trigger);
  fflush(stdout);


  if (result != N) {trigger = 0;}

  printf("trigger = %d after setting\n", trigger);
  fflush(stdout);

  return trigger;

}

} //namespace KokkosResilience
 */

#endif //defined(KOKKOS_ENABLE_OPENMP) //&& defined (KR_ENABLE_ACTIVE_EXECUTION_SPACE)
#endif