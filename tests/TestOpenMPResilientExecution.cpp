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
#include <gtest/gtest.h>
#include "TestCommon.hpp"
#include <Kokkos_Core.hpp>
#include <Kokkos_Core_fwd.hpp>
#include <resilience/Resilience.hpp>
#include <resilience/openMP/ResHostSpace.hpp>
#include <resilience/openMP/ResOpenMP.hpp>

#include <thread>
#include <vector>
#include <omp.h>
#include <cstdio>

#define N 20
#define N_2 10
#define MemSpace KokkosResilience::ResHostSpace
#define ExecSpace KokkosResilience::ResOpenMP

//Resilient
using range_policy = Kokkos::RangePolicy<ExecSpace>;
using ViewVectorIntSubscriber = Kokkos::View< int* , Kokkos::LayoutRight, MemSpace,
        Kokkos::Experimental::SubscribableViewHooks<
                KokkosResilience::ResilientDuplicatesSubscriber > >;
using ViewVectorDoubleSubscriber = Kokkos::View< double* , Kokkos::LayoutRight, MemSpace,
        Kokkos::Experimental::SubscribableViewHooks<
                KokkosResilience::ResilientDuplicatesSubscriber > >;
using ConstViewVectorDoubleSubscriber = Kokkos::View< const double*, Kokkos::LayoutRight, MemSpace,
        Kokkos::Experimental::SubscribableViewHooks<
                KokkosResilience::ResilientDuplicatesSubscriber > >;

//Non-resilient
using ViewVectorType = Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::HostSpace>;
using range_policy2 = Kokkos::RangePolicy<Kokkos::OpenMP>;

/*********************************
*********PARALLEL FORS************
**********************************/

// gTest runs parallel_for with non-resilient Kokkos. Should never fail.
TEST(TestResOpenMP, TestKokkosFor)
{
  // Allocate y, x vectors.
  ViewVectorType y2( "y", N );
  ViewVectorType x2( "x", N );

  Kokkos::Timer timer;
  // Initialize y vector on host using parallel_for
  Kokkos::parallel_for(
      range_policy2(0, N), KOKKOS_LAMBDA(int i) { y2(i) = i; });

  Kokkos::deep_copy(x2, y2);
  for ( int i = 0; i < N; i++) {
    ASSERT_EQ(x2(i), i);
  }
}

// gTest runs parallel_for with resilient Kokkos doubles assignment and atomic counter.
// Expect counter to count iterations.
TEST(TestResOpenMP, TestResilientForDouble)
{
  // Allocate y, x vectors.
  ViewVectorDoubleSubscriber y( "y", N );
  ViewVectorDoubleSubscriber x( "x", N );

  //Integer vector 1 long to count data accesses, because scalar view bugs (previously)
  ViewVectorIntSubscriber counter( "DataAccesses", 1);

  Kokkos::Timer timer;

  counter(0) = 0;

  //Initialize y vector on host using parallel_for, increment a counter for data accesses.
  Kokkos::parallel_for( range_policy (0, N), KOKKOS_LAMBDA ( const int i) {
    y ( i ) = i;
    Kokkos::atomic_increment(&counter(0));
  });

  KokkosResilience::clear_duplicates_cache();

  Kokkos::deep_copy(x, y);
  for ( int i = 0; i < N; i++) {
    ASSERT_EQ(x(i), i);
  }

  ASSERT_EQ(counter(0), N);
}

// gTest runs parallel_for with resilient Kokkos integer assignment and atomic counter.
// Expect counter to count iterations.
TEST(TestResOpenMP, TestResilientForInteger)
{
  // Allocate y, x vectors.
  ViewVectorIntSubscriber  y( "y", N );
  ViewVectorIntSubscriber  x( "x", N );

  //Integer vector 1 long to count data accesses, because scalar view bugs (previously)
  ViewVectorIntSubscriber  counter( "DataAccesses", 1);

  Kokkos::Timer timer;

  counter(0) = 0;

  //Initialize y vector on host using parallel_for, increment a counter for data accesses.
  Kokkos::parallel_for( range_policy (0, N), KOKKOS_LAMBDA ( const int i) {
    y ( i ) = i;
    Kokkos::atomic_increment(&counter(0));
  });

  Kokkos::deep_copy(x, y);
  for ( int i = 0; i < N; i++) {
    ASSERT_EQ(x(i), i);
  }
  KokkosResilience::clear_duplicates_cache();

  ASSERT_EQ(counter(0), N);
}

// gTest attempts to trigger all 3 executions generating different data.
// Requires non-multipe of 3 OMP threads to generate error.
// Should repeat user-specified number of times (in context file) and then abort.
TEST(TestResOpenMP, TestResilientForInsertError)
{

  ViewVectorIntSubscriber counter ( "DataAccesses", 1);

  // Allocate y, x vectors.
  ViewVectorDoubleSubscriber y( "y", N );
  ViewVectorDoubleSubscriber x( "x", N );

  counter(0) = 0;

  ::testing::FLAGS_gtest_death_test_style = "threadsafe";

  // Assigning each y(i) threadId, should cause a failure in the resilient execution except in single-thread case.
  EXPECT_DEATH(
    Kokkos::parallel_for( range_policy (0, N), KOKKOS_LAMBDA ( const int i) {
      y(i) = omp_get_thread_num();
      Kokkos::atomic_increment(&counter(0));
    });
  ,"Aborted in parallel_for, resilience majority voting failed because each execution obtained a differing value.");
}

// gTest runs parallel_for with resilient Kokkos doubles assignment and atomic counter.
// Uses a non-zero range start
TEST(TestResOpenMP, TestResilientNonZeroRange)
{

  // Allocate y, x vectors.
  ViewVectorDoubleSubscriber y( "y", N );
  ViewVectorDoubleSubscriber x( "x", N );

  //Initialize y vector on host using parallel_for, increment a counter for data accesses.
  Kokkos::parallel_for( range_policy (0, N_2), KOKKOS_LAMBDA ( const int i) {
    y ( i ) = 1;
  });

  Kokkos::parallel_for( range_policy (N_2, N), KOKKOS_LAMBDA ( const int i) {
    y ( i ) = 500;
  });

  Kokkos::deep_copy(x, y);

  KokkosResilience::clear_duplicates_cache();

  for ( int i = 0; i < N; i++) {
    if (i<N_2) {
      ASSERT_EQ(x(i), 1);
    } else {
      ASSERT_EQ (x(i), 500);
    }
  }

}

// Test runs parallel_for with a const view. Expect const view to trigger const view subscriber, a no-op
// Expect non-const view to trigger copies and majority voting resiliency subscriber
TEST(TestResOpenMP, TestConstViewSubscriber)
{

  ViewVectorDoubleSubscriber x( "x", N );
  ViewVectorDoubleSubscriber y( "y", N );

  Kokkos::parallel_for( range_policy (0, N), KOKKOS_LAMBDA ( const int i) {
    x ( i ) = i;
  });

  ConstViewVectorDoubleSubscriber x_const = x;

  Kokkos::parallel_for( range_policy (0, N), KOKKOS_LAMBDA ( const int i) {
    y ( i ) = 2 * x_const (i);
  });

  KokkosResilience::clear_duplicates_cache();

  for ( int i = 0; i < N; i++) {
    ASSERT_EQ(y(i), 2 * i);
  }

}
