#include <gtest/gtest.h>
#include "TestCommon.hpp"
#include <Kokkos_Core.hpp>
#include <Kokkos_Core_fwd.hpp>
#include <resilience/Resilience.hpp>
#include <resilience/openMP/ResHostSpace.hpp>
#include <resilience/openMP/ResOpenMP.hpp>
//#include <resilience/openMP/OpenMPResSubscriber.hpp>

#include <ctime>
#include <random>
#include <time.h>
#include <thread>

#include <vector>
#include <math.h>

// included for thread prints
#include <omp.h>

//#ifdef KR_ENABLE_OPENMP // TODO: RESILIENCE HEADER MODIFY
//#ifdef KR_ENABLE_RESILIENT_EXECUTION_SPACE // TODO: REIMPLEMENT

#define N 25
#define MemSpace KokkosResilience::ResHostSpace
#define ExecSpace KokkosResilience::ResOpenMP

TEST(TestResOpenMP, TestThreads)
{
  int max = omp_get_max_threads();
  std::cout << "Max threads: " << max << std::endl;

#pragma omp parallel
  {
    int n = omp_get_num_threads();
    int tid = omp_get_thread_num();
    std::cout << "There are " << n  << " threads: Hello from thread: " << tid << std::endl;
  };
  ASSERT_EQ( omp_get_max_threads(), 8);
}

// gTest resilient spaces work on own. Goal is to deepcopy one view to another.
TEST(TestResOpenMP, TestSpaces)
{
  Kokkos::fence();

  using range_policy = Kokkos::RangePolicy<ExecSpace>;

  using ViewVectorType = Kokkos::View<double*, Kokkos::LayoutRight, MemSpace>;

  ViewVectorType y( "y", N);
  ViewVectorType x( "x", N);

  for ( int i = 0; i < N; i++ ) {
    y( i ) = 1;
  }

  Kokkos::Timer timer;
  Kokkos::deep_copy(x, y);
  double time = timer.seconds();

  std::cout << "The deep-copy took " << time << " seconds." << std::endl;

  for ( int i = 0; i < N; i++) {
    ASSERT_EQ(x(i), 1);
  }

  printf("\n\n\n");
  fflush(stdout);

}

// gTest resilient spaces work on own. Goal is to deepcopy one view to another.
TEST(TestResOpenMP, TestDuplicateSubscriber)
{
  Kokkos::fence();

  using range_policy = Kokkos::RangePolicy<ExecSpace>;

  using ViewVectorType = Kokkos::View<double*, Kokkos::LayoutRight, MemSpace>;

  ViewVectorType y( "y", N);
  ViewVectorType x( "x", N);

  for ( int i = 0; i < N; i++ ) {
    y( i ) = 1;
  }

  Kokkos::Timer timer;
  Kokkos::deep_copy(x, y);
  double time = timer.seconds();

  std::cout << "The deep-copy took " << time << " seconds." << std::endl;

  for ( int i = 0; i < N; i++) {
    ASSERT_EQ(x(i), 1);
  }

  printf("\n\n\n");
  fflush(stdout);

}

/*********************************
*********PARALLEL FORS************
**********************************/

// gTest runs parallel_for with non-resilient Kokkos. Should never fail.
TEST(TestResOpenMP, TestKokkosFor)
{

  std::cout << "KokkosFor Test Line 1" << std::endl;
  using range_policy2 = Kokkos::RangePolicy<Kokkos::OpenMP>;
  using ViewVectorType2 = Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::HostSpace>;

  // Allocate y, x vectors.
  ViewVectorType2 y2( "y", N );
  ViewVectorType2 x2( "x", N );
  std::cout << "KokkosFor Test Line 2" << std::endl;
  Kokkos::Timer timer;

  Kokkos::fence();

  auto test(y2);


  // Initialize y vector on host using parallel_for
  Kokkos::parallel_for( range_policy2 (0, N), KOKKOS_LAMBDA ( int i) {
    y2 ( i ) = i;
  });
  std::cout << "KokkosFor Test Line 3" << std::endl;
  Kokkos::fence();
  double time = timer.seconds();

  Kokkos::deep_copy(x2, y2);
  for ( int i = 0; i < N; i++) {
    ASSERT_EQ(x2(i), i);
  }

  printf("GTEST: Thread %d reports Kokkos parallel_for took %f seconds.\n", omp_get_thread_num(), time);

  printf("\n\n\n");
  fflush(stdout);

}


// gTest runs parallel_for with resilient Kokkos. Expect same answer as last test.
TEST(TestResOpenMP, TestResilientFor)
{

  // range policy with resilient execution space
  using range_policy = Kokkos::RangePolicy<ExecSpace>;

  // test vector types with the duplicating subscriber
  using subscriber_vector_double_type = Kokkos::View< double* , Kokkos::LayoutRight, MemSpace,
                                                      Kokkos::Experimental::SubscribableViewHooks<
                                                          KokkosResilience::ResilientDuplicatesSubscriber > >;
  using subscriber_vector_int_type = Kokkos::View< int* , Kokkos::LayoutRight, MemSpace,
                                                   Kokkos::Experimental::SubscribableViewHooks<
                                                       KokkosResilience::ResilientDuplicatesSubscriber > >;

  // Allocate scalar for test incrementation
  //typedef Kokkos::View<int, Kokkos::LayoutRight, MemSpace> ViewScalarInt;
  //ViewScalarInt set_data_counter;
  //set_data_counter() = 1;

  //Integer vector 1 long to count data accesses, because scalar view bugs (previously)
  subscriber_vector_int_type counter( "DataAccesses", 1);

  // Allocate y, x vectors.
  subscriber_vector_double_type y( "y", N );
  subscriber_vector_double_type x( "x", N );

  printf("GTEST: Thread %d reports vectors declared.\n", omp_get_thread_num());
  fflush(stdout);

  Kokkos::Timer timer;

  counter(0) = 0;

  printf("GTEST: Thread %d reports counter successfully initialized to %d.\n", omp_get_thread_num(), counter(0));
  fflush(stdout);

  //Initialize y vector on host using parallel_for, increment a counter for data accesses.
  Kokkos::parallel_for( range_policy (0, N), KOKKOS_LAMBDA ( const int i) {
    y ( i ) = i;
    Kokkos::atomic_increment(&counter(0));
  });

  Kokkos::fence();

  printf("GTEST: Thread %d reports test parallel_for completed, accuracy TBD.\n", omp_get_thread_num());
  fflush(stdout);
  printf("GTEST: Thread %d reports counter is %d. It should be %d.\n", omp_get_thread_num(), counter(0), N);
  fflush(stdout);

  double time = timer.seconds();

  Kokkos::deep_copy(x, y);

  for ( int i = 0; i < N; i++) {
    ASSERT_EQ(x(i), i);
  }

  printf("GTEST: Thread %d reports test parallel_for completed. Data assignment was correct.\n", omp_get_thread_num());
  fflush(stdout);

  ASSERT_EQ(counter(0), N);

  printf("\n\n\n");
  fflush(stdout);
}

/*
// gTest attempts to trigger all 2 executions generating different data. 
// Should repeat user-specified number of times (in context file) and then abort.
TEST(TestResOpenMP, TestResilientForInsertError)
{

  using range_policy = Kokkos::RangePolicy<ExecSpace>;

  typedef Kokkos::View<int*, Kokkos::LayoutRight, MemSpace> ViewVectorInt;
  ViewVectorInt counter( "DataAccesses", 1);

  // Allocate y, x vectors.
  typedef Kokkos::View<double*, Kokkos::LayoutRight, MemSpace>   ViewVectorType;
  ViewVectorType y( "y", N );
  ViewVectorType x( "x", N );

  counter(0) = 0;

  printf("GTEST: Thread %d reports counter successfully initialized to %d.\n", omp_get_thread_num(), counter(0));
  fflush(stdout);

  ::testing::FLAGS_gtest_death_test_style = "threadsafe";

  //TODO: TEST EXPECTED FAIL NOT FAILING
  // Assigning each y(i) threadId, should cause a failure in the resilient execution.
  EXPECT_DEATH(
  Kokkos::parallel_for( range_policy (0, N), KOKKOS_LAMBDA ( const int i) {
    y(i) = omp_get_thread_num();
    std::cout << y(i) << std::endl;
    Kokkos::atomic_increment(&counter(0));
  });

  , "GTEST EXPECT_DEATH: Kokkos parallel_for failed in OMP assign loop");

  printf("\n\n\n");
  fflush(stdout);
}
*/
