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

#include <omp.h>

//#ifdef KR_ENABLE_OPENMP

#define N 25
#define MemSpace KokkosResilience::ResHostSpace
#define ExecSpace KokkosResilience::ResOpenMP

// gTest resilient spaces work on own. Goal is to deepcopy one view to another.
TEST(TestResOpenMP, TestSpaces)
{
  Kokkos::fence();

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

  std::cout << std::endl << std::endl;

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

  Kokkos::Timer timer;

  Kokkos::fence();

  // Initialize y vector on host using parallel_for
  Kokkos::parallel_for( range_policy2 (0, N), KOKKOS_LAMBDA ( int i) {
    y2 ( i ) = i;
  });

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

  // Allocate y, x vectors.
  subscriber_vector_double_type y( "y", N );
  subscriber_vector_double_type x( "x", N );

  printf("GTEST: Thread %d reports vectors declared.\n", omp_get_thread_num());
  fflush(stdout);

  Kokkos::Timer timer;

  //Initialize y vector on host using parallel_for
  Kokkos::parallel_for( range_policy (0, N), KOKKOS_LAMBDA ( const int i) {
    y ( i ) = i;
    //printf("This is i from the parallel for %d \n", i);
  });

  Kokkos::fence();

  printf("GTEST: Thread %d reports test parallel_for completed, accuracy TBD.\n", omp_get_thread_num());
  fflush(stdout);

  double time = timer.seconds();

  Kokkos::deep_copy(x, y);

  for ( int i = 0; i < N; i++) {
    ASSERT_EQ(x(i), i);
  }

  printf("GTEST: Thread %d reports test parallel_for completed. Data assignment was correct.\n", omp_get_thread_num());
  fflush(stdout);
  printf("\n\n\n");
  fflush(stdout);
}

// gTest runs parallel_for with resilient Kokkos and atomic counter.
// Expect counter to count iterations.
TEST(TestResOpenMP, TestResilientForAtomics)
{
  // range policy with resilient execution space
  using range_policy = Kokkos::RangePolicy<ExecSpace>;

  using subscriber_vector_int_type = Kokkos::View< int* , Kokkos::LayoutRight, MemSpace,
      Kokkos::Experimental::SubscribableViewHooks<
          KokkosResilience::ResilientDuplicatesSubscriber > >;

  //Integer vector 1 long to count data accesses, because scalar view bugs (previously)
  subscriber_vector_int_type counter( "DataAccesses", 1);

  Kokkos::Timer timer;

  counter(0) = 0;

  printf("GTEST: Thread %d reports counter successfully initialized to %d.\n", omp_get_thread_num(), counter(0));
  fflush(stdout);

  //Initialize y vector on host using parallel_for, increment a counter for data accesses.
  Kokkos::parallel_for( range_policy (0, N), KOKKOS_LAMBDA ( const int i) {
    //printf("This is i from the parallel for %d \n", i);
    Kokkos::atomic_increment(&counter(0));
    //printf("This is counter(0): %d\n", counter(0));
    //fflush(stdout);
  });

  Kokkos::fence();

  printf("GTEST: Thread %d reports counter is %d. It should be %d.\n", omp_get_thread_num(), counter(0), N);
  fflush(stdout);

  double time = timer.seconds();

  ASSERT_EQ(counter(0), N);

  printf("\n\n\n");
  fflush(stdout);
}

// gTest runs parallel_for with resilient Kokkos  assignment and atomic counter.
// Expect counter to count iterations.
TEST(TestResOpenMP, TestResilientForAtomicsWithAssignment)
{
  // range policy with resilient execution space
  using range_policy = Kokkos::RangePolicy<ExecSpace>;

  using subscriber_vector_int_type = Kokkos::View< int* , Kokkos::LayoutRight, MemSpace,
      Kokkos::Experimental::SubscribableViewHooks<
          KokkosResilience::ResilientDuplicatesSubscriber > >;
  using subscriber_vector_double_type = Kokkos::View< double* , Kokkos::LayoutRight, MemSpace,
      Kokkos::Experimental::SubscribableViewHooks<
          KokkosResilience::ResilientDuplicatesSubscriber > >;

  // Allocate y, x vectors.
  subscriber_vector_double_type y( "y", N );
  subscriber_vector_double_type x( "x", N );

  printf("GTEST: Thread %d reports vectors declared.\n", omp_get_thread_num());
  fflush(stdout);

  //Integer vector 1 long to count data accesses, because scalar view bugs (previously)
  subscriber_vector_int_type counter( "DataAccesses", 1);

  Kokkos::Timer timer;

  counter(0) = 0;

  printf("GTEST: Thread %d reports counter successfully initialized to %d.\n", omp_get_thread_num(), counter(0));
  fflush(stdout);

  //Initialize y vector on host using parallel_for, increment a counter for data accesses.
  Kokkos::parallel_for( range_policy (0, N), KOKKOS_LAMBDA ( const int i) {
    y ( i ) = i;
    //printf("This is i from the parallel for %d \n", i);
    //fflush(stdout);
    Kokkos::atomic_increment(&counter(0));
    //printf("This is counter(0): %d\n", counter(0));
    //fflush(stdout);

  });

  Kokkos::fence();

  double time = timer.seconds();
  printf("GTEST: Thread %d reports counter is %d. It should be %d.\n", omp_get_thread_num(), counter(0), N);
  fflush(stdout);

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

// gTest attempts to trigger all 3 executions generating different data.
// Should repeat user-specified number of times (in context file) and then abort.
TEST(TestResOpenMP, TestResilientForInsertError)
{

  using range_policy = Kokkos::RangePolicy<ExecSpace>;

  using subscriber_vector_int_type = Kokkos::View< int* , Kokkos::LayoutRight, MemSpace,
  Kokkos::Experimental::SubscribableViewHooks<
  KokkosResilience::ResilientDuplicatesSubscriber > >;
  using subscriber_vector_double_type = Kokkos::View< double* , Kokkos::LayoutRight, MemSpace,
  Kokkos::Experimental::SubscribableViewHooks<
  KokkosResilience::ResilientDuplicatesSubscriber > >;

  subscriber_vector_int_type counter ( "DataAccesses", 1);

  // Allocate y, x vectors.
  subscriber_vector_double_type y( "y", N );
  subscriber_vector_double_type x( "x", N );

  counter(0) = 0;

  printf("GTEST: Thread %d reports counter successfully initialized to %d.\n", omp_get_thread_num(), counter(0));
  fflush(stdout);

  ::testing::FLAGS_gtest_death_test_style = "threadsafe";

  // Assigning each y(i) threadId, should cause a failure in the resilient execution.
  EXPECT_DEATH(
  Kokkos::parallel_for( range_policy (0, N), KOKKOS_LAMBDA ( const int i) {
    y(i) = omp_get_thread_num();
    //std::cout << y(i) << std::endl;
    Kokkos::atomic_increment(&counter(0));
  });
,"Aborted in parallel_for, resilience majority voting failed because each execution obtained a differing value.");

  printf("\n\n\n");
  fflush(stdout);
}

