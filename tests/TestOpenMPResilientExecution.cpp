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

//#ifdef KR_ENABLE_OPENMP

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
  std::cout << "KokkosFor Test" << std::endl;

  // Allocate y, x vectors.
  ViewVectorType y2( "y", N );
  ViewVectorType x2( "x", N );

  Kokkos::Timer timer;
  // Initialize y vector on host using parallel_for
  Kokkos::parallel_for( range_policy2 (0, N), KOKKOS_LAMBDA ( int i) {
    y2 ( i ) = i;
  });
  double time = timer.seconds();

  Kokkos::deep_copy(x2, y2);
  for ( int i = 0; i < N; i++) {
    ASSERT_EQ(x2(i), i);
  }

  std::cout << "GTEST: Thread " << omp_get_thread_num() << " reports Kokkos parallel_for took " << time << " seconds." << std::endl;
  std::cout << std::endl;
  std::cout << std::endl;
  std::cout << std::endl;

}

// gTest runs parallel_for with resilient Kokkos doubles assignment and atomic counter.
// Expect counter to count iterations.
TEST(TestResOpenMP, TestResilientForDouble)
{
  std::cout << "KokkosResilient For Doubles" << std::endl;

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
  double time = timer.seconds();
  std::cout << "GTEST: Thread " << omp_get_thread_num() << " reports Kokkos parallel_for took " << time << " seconds." << std::endl;
  std::cout << "GTEST: Thread " << omp_get_thread_num() << " reports counter is " << counter(0) << ". It should be " << N << "." << std::endl;

  Kokkos::deep_copy(x, y);
  for ( int i = 0; i < N; i++) {
    ASSERT_EQ(x(i), i);
  }

  std::cout << "GTEST: Thread " << omp_get_thread_num() << " reports test parallel_for completed. Data assignment was correct." << std::endl;

  ASSERT_EQ(counter(0), N);

  std::cout << std::endl;
  std::cout << std::endl;
  std::cout << std::endl;

}

// gTest runs parallel_for with resilient Kokkos integer assignment and atomic counter.
// Expect counter to count iterations.
TEST(TestResOpenMP, TestResilientForInteger)
{
  std::cout << "KokkosResilient For Integers" << std::endl;

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
  double time = timer.seconds();
  std::cout << "GTEST: Thread " << omp_get_thread_num() << " reports Kokkos parallel_for took " << time << " seconds." << std::endl;
  std::cout << "GTEST: Thread " << omp_get_thread_num() << " reports counter is " << counter(0) << ". It should be " << N << "." << std::endl;

  Kokkos::deep_copy(x, y);
  for ( int i = 0; i < N; i++) {
    ASSERT_EQ(x(i), i);
  }
  KokkosResilience::clear_duplicates_cache();

  std::cout << "GTEST: Thread " << omp_get_thread_num() << " reports test parallel_for completed. Data assignment was correct." << std::endl;

  ASSERT_EQ(counter(0), N);

  std::cout << std::endl;
  std::cout << std::endl;
  std::cout << std::endl;
}

// gTest attempts to trigger all 3 executions generating different data.
// Requires non-multipe of 3 OMP threads to generate error.
// Should repeat user-specified number of times (in context file) and then abort.
TEST(TestResOpenMP, TestResilientForInsertError)
{
  std::cout << "KokkosResilient For OMP Thread Error" << std::endl;

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
  std::cout << "KokkosResilient NonZeroRange" << std::endl;

  // Allocate y, x vectors.
  ViewVectorDoubleSubscriber y( "y", N );
  ViewVectorDoubleSubscriber x( "x", N );

  //Initialize y vector on host using parallel_for, increment a counter for data accesses.
  Kokkos::parallel_for( range_policy (0, N_2), KOKKOS_LAMBDA ( const int i) {
    y ( i ) = 1;
  /*  if (i==1) {
      std::cout << "Threads:" << omp_get_num_threads();
      std::cout << std::endl;
    }*/
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

  std::cout << "GTEST: Thread " << omp_get_thread_num() << " reports test parallel_for completed. Data assignment was correct." << std::endl;
  std::cout << std::endl;
  std::cout << std::endl;
  std::cout << std::endl;
}

// Test runs parallel_for with a const view. Expect const view to trigger const view subscriber, a no-op
// Expect non-const view to trigger copies and majority voting resiliency subscriber
TEST(TestResOpenMP, TestConstViewSubscriber)
{
  std::cout << "KokkosResilient Test Constant View Subscriber" << std::endl;

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

  std::cout << "Success! The computation was correct! View debug info for constant view copying." << std::endl;
  std::cout << std::endl << std::endl << std::endl;
}

// gTest runs parallel_reduce with regular Kokkos to get a dot product. Should never fail.
TEST(TestResOpenMP, TestKokkosReduceDouble)
{
  std::cout << "Kokkos parallel_reduce dot product test." << std::endl;

  ViewVectorType y ( "y", N );
  ViewVectorType x ( "x", N );

  Kokkos::Timer timer;

  //Initialize y vector on host using parallel_for, increment a counter for data accesses.
  Kokkos::parallel_for( range_policy2 (0, N), KOKKOS_LAMBDA ( const int i) {
    y ( i ) = 1.0;
  });

  Kokkos::deep_copy(x,y);

  double dot_product = 0;

  Kokkos::parallel_reduce(range_policy2(0,N),KOKKOS_LAMBDA (const int i, double & update) {
    update += x ( i ) * y ( i );
  }, dot_product);

  double time = timer.seconds();

  KokkosResilience::clear_duplicates_cache();

  std::cout << "GTEST: Thread " << omp_get_thread_num() << " reports parallel_reduce completed in time " << time << "." << std::endl;
  std::cout << "Dot product was " << dot_product << " and should have been " << N << "." << std::endl;
  std::cout << std::endl;
  std::cout << std::endl;
  std::cout << std::endl;

  ASSERT_EQ(dot_product, N);

}

// gTest runs parallel_reduce with resilient Kokkos to get a dot product.
TEST(TestResOpenMP, TestResilientReduceDouble)
{
  std::cout << "Kokkos resilient parallel_reduce dot product test." << std::endl;

  // Allocate y, x vectors.
  ViewVectorDoubleSubscriber y( "y", N );
  ViewVectorDoubleSubscriber x( "x", N );

  Kokkos::Timer timer;

  //Initialize y vector on host using parallel_for, increment a counter for data accesses.
  Kokkos::parallel_for( range_policy (0, N), KOKKOS_LAMBDA ( const int i) {
    y ( i ) = 1.0;
  });

  Kokkos::deep_copy(x,y);
  double dot_product = 0;

  Kokkos::parallel_reduce(range_policy(0,N),KOKKOS_LAMBDA (const int i, double & update) {
    update += x ( i ) * y ( i );
  }, dot_product);

  double time = timer.seconds();

  KokkosResilience::clear_duplicates_cache();

  std::cout << "GTEST: Thread " << omp_get_thread_num() << " reports parallel_reduce completed in time " << time << "." << std::endl;
  std::cout << "Dot product was " << dot_product << " and should have been " << N << "." << std::endl;
  std::cout << std::endl;
  std::cout << std::endl;
  std::cout << std::endl;

  ASSERT_EQ(dot_product, N);

}