#include <gtest/gtest.h>
#include "TestCommon.hpp"
#include <Kokkos_Core.hpp>
#include <Kokkos_Core_fwd.hpp>
#include <resilience/Resilience.hpp>
//#include <resilience/OpenMP/ResHostSpace.hpp>
//#include <resilience/OpenMP/ResOpenMP.hpp>

#include <ctime>
#include <random>
#include <time.h>
#include <thread>

//#include <gsl/gsl_randist.h>
//#include <gsl/gsl_rng.h>
#include <vector>
#include <math.h>

// included for thread prints
#include <omp.h>

//#ifdef KR_ENABLE_OPENMP // TODO: RESILIENCE HEADER MODIFY
//#ifdef KR_ENABLE_RESILIENT_EXECUTION_SPACE // TODO: REIMPLEMENT

#define N 25
#define MemSpace KokkosResilience::ResHostSpace
#define ExecSpace KokkosResilience::ResOpenMP

// gTest resilient spaces work on own. Goal is to deepcopy one view to another.
TEST(TestResOpenMP, TestSpaces)
{

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
  using range_policy = Kokkos::RangePolicy<Kokkos::OpenMP>;

  // Allocate y, x vectors.
  typedef Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::HostSpace> ViewVectorType;
  ViewVectorType y( "y", N );
  ViewVectorType x( "x", N );

  Kokkos::Timer timer;

  // Initialize y vector on host using parallel_for
  Kokkos::parallel_for( range_policy (0, N), KOKKOS_LAMBDA ( const int i) {
    y ( i ) = i;
  });
 
  Kokkos::fence();
  double time = timer.seconds();
  
  Kokkos::deep_copy(x, y);
  for ( int i = 0; i < N; i++) {
    ASSERT_EQ(x(i), i);
  }

  printf("GTEST: Thread %d reports Kokkos parallel_for took %f seconds.\n", omp_get_thread_num(), time);

  printf("\n\n\n");
  fflush(stdout);

}
 
// gTest runs parallel_for with resilient Kokkos. Expect same answer as last test.
TEST(TestResOpenMP, TestResilientFor)
{

  using range_policy = Kokkos::RangePolicy<ExecSpace>;

  // Allocate scalar for test incrementation
  //typedef Kokkos::View<int, Kokkos::LayoutRight, MemSpace> ViewScalarInt;
  //ViewScalarInt set_data_counter;

  //set_data_counter() = 1;
  
  // Fix with integer vector type for now
  typedef Kokkos::View<int*, Kokkos::LayoutRight, MemSpace> ViewVectorInt;
  ViewVectorInt counter( "DataAccesses", 1);

  // Allocate y, x vectors.
  typedef Kokkos::View<double*, Kokkos::LayoutRight, MemSpace>   ViewVectorType;
  ViewVectorType y( "y", N );
  ViewVectorType x( "x", N );
  
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


/**********************************
********PARALLEL REDUCES***********
**********************************/
/*
// gTest runs parallel_reduce with non-resilient Kokkos. Should never fail.
TEST(TestResOpenMP, TestKokkosReduce)
{
 
  using range_policy = Kokkos::RangePolicy<Kokkos::OpenMP>;

  // Allocate y, x vectors.
  typedef Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::HostSpace>   ViewVectorType;
  ViewVectorType y( "y", N );
  ViewVectorType x( "x", N );

  printf("GTEST: Thread %d reports vectors declared.\n", omp_get_thread_num());
  fflush(stdout);

  double result = 0;
  double correct = N;

  // Initialize y vector on host using parallel_for
  Kokkos::parallel_for( range_policy (0, N), KOKKOS_LAMBDA ( const int i) {
    y ( i ) = 1;
  });

  printf("GTEST: Thread %d reports vector y initialized.\n", omp_get_thread_num());
  fflush(stdout);

  Kokkos::fence();
  
  Kokkos::deep_copy(x, y);
  
  printf("GTEST: Thread %d reports vector y deep-copied to x.\n", omp_get_thread_num());
  fflush(stdout);

  Kokkos::Timer timer;

  // Perform vector dot product y*x using parallel_reduce
  Kokkos::parallel_reduce( "yx", N, KOKKOS_LAMBDA ( int j, double &update ) {
    update += y( j ) * x( j );
  }, result );

  Kokkos::fence();

  printf("GTEST: Thread %d reports parallel_reduce finished.\n", omp_get_thread_num());
  fflush(stdout);

  // Calculate time.
  double time = timer.seconds(); 
 
  printf("It took %f seconds to perform the parallel_reduce.y\n", time);
  fflush(stdout);
  printf("The correct length of two all ones vectors multiplied together is N. N is %f.\n", correct);
  fflush(stdout);
  printf("The result from parallel_reduce was %f. \n", result);
  fflush(stdout);
  
  printf("\n\n\n");
  fflush(stdout);

  ASSERT_EQ(result, correct);

}

// gTest runs parallel_reduce with resilient Kokkos. Expect same answer as last test.
TEST(TestResOpenMP, TestParallelReduce)
{
 
  using range_policy = Kokkos::RangePolicy<ExecSpace>;
  //Kokkos::RangePolicy<KokkosResilience::ResOpenMP>
  
  // Allocate y, x vectors.
  typedef Kokkos::View<double*, Kokkos::LayoutRight, MemSpace> ViewVectorType;
  ViewVectorType y( "y", N );
  ViewVectorType x( "x", N );

  printf("GTEST: Thread %d reports vectors declared.\n", omp_get_thread_num());
  fflush(stdout);

  double result = 0;
  double correct = N;

  // Initialize y vector on host using regular for
  for ( int i = 0; i < N; i++ ) {
    y( i ) = 1;
  }

  Kokkos::fence();

  printf("GTEST: Thread %d reports y initialized.\n", omp_get_thread_num());
  fflush(stdout);
  
  Kokkos::deep_copy(x, y);

  Kokkos::fence();

  printf("GTEST: Thread %d reports y deep-copied to x.\n", omp_get_thread_num());
  fflush(stdout);

  Kokkos::Timer timer;

  // Perform vector dot product y*x using parallel_reduce
  //Kokkos::RangePolicy<KokkosResilience::ResOpenMP> range_policy = Kokkos::RangePolicy<KokkosResilience::ResOpenMP>(0,N);

  Kokkos::parallel_reduce( "yx", range_policy(0,N), KOKKOS_LAMBDA ( int j, double &update ) {
    update += y( j ) * x( j );
  }, result );

  Kokkos::fence();

  printf("GTEST: Thread %d reports parallel_reduce finished.\n", omp_get_thread_num());
  fflush(stdout);

  double time = timer.seconds(); 
 
  printf("It took %f seconds to perform the parallel_reduce.\n", time);
  fflush(stdout);
  printf("The correct length of two all ones vectors multiplied together is N. N is %f.\n", correct);
  fflush(stdout);
  printf("The result from parallel_reduce was %f. \n", result);
  fflush(stdout);
  
  printf("\n\n\n");
  fflush(stdout);

  ASSERT_EQ(result, correct);

}
*/
/**********************************
**********PARALLEL SCANS***********
**********************************/
/*
// gTest if the ParallelScan test works with regular Kokkos.
TEST(TestResOpenMP, TestRegularScan)
{
 
  using range_policy = Kokkos::RangePolicy<Kokkos::OpenMP>;

  // Allocate x vector.
  typedef Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::HostSpace>   ViewVectorType;
  ViewVectorType x( "x", N );

  // Initialize x vector on host using parallel_for
  Kokkos::parallel_for( range_policy (0, N), KOKKOS_LAMBDA ( const int i) {
    x ( i ) = 1.0;
  });
 
  Kokkos::fence();
 
  // Timer
  Kokkos::Timer timer;

  // Perform vector element sum using parallel_scan
  Kokkos::parallel_scan( "xscan", N, KOKKOS_LAMBDA ( int j, double &update, const bool& final ) {
    update += x( j );
    //printf("Update: %f \n", update);
    //fflush(stdout);
    if(final){
      x( j ) = update;
    }
  });

  Kokkos::fence();

  // Calculate time.
  double time = timer.seconds(); 

  printf("It took %f seconds to perform the parallel_for, not including deep-copy\n", time);
  fflush(stdout);
 
  // TESTING SCAN VALUES: scan(0) = x(0), scan(2) = (3), scan(n-1) = N
  printf("Correct Scan(0) = %f, Parallel_Scan(0) = %f \n", 1, x((int)0));
  fflush(stdout);
  printf("Correct Scan(2) = %f, Parallel_Scan(2) = %f \n", 3, x(2));
  fflush(stdout);  
  printf("Correct Scan(N-1) = %f, Parallel_Scan(N-1) = %f \n", N, x(N-1));
  fflush(stdout);  

  ASSERT_EQ(x(0), 1);
  ASSERT_EQ(x(2), 3);
  ASSERT_EQ(x(N-1), N);

}


// gTest if the parallel_scan test works with resilient Kokkos.
TEST(TestResOpenMP, TestParallelScan)
{
 
  using range_policy = Kokkos::RangePolicy<ExecSpace>;

  // Allocate y vector.
  typedef Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::HostSpace>   ViewVectorType;
  ViewVectorType y( "y", N );

  // Initialize x vector on host using regular for (isolate parallel_scan)
  for ( int i = 0; i < N; i++ ) {
    y( i ) = 1.0;
  }
   
  // Timer
  Kokkos::Timer timer;

  // Perform vector element sum using parallel_scan
  Kokkos::parallel_scan( "yscan", N, KOKKOS_LAMBDA ( int j, double &update, const bool& final ) {
    update += y( j );
   // printf("Update: %f \n", update);
   // fflush(stdout);
    if(final){
      y( j ) = update;
    }
  });

  Kokkos::fence();

  // Calculate time.
  double time = timer.seconds(); 

  printf("It took %f seconds to perform the parallel_for, not including deep-copy\n", time);
  fflush(stdout);
 
  // TESTING SCAN VALUES: scan(0) = x(0), scan(2) = (3), scan(n-1) = N
  printf("Correct Scan(0) = %f, Parallel_Scan(0) = %f \n", 1, y(0));
  fflush(stdout);
  printf("Correct Scan(2) = %f, Parallel_Scan(2) = %f \n", 3, y(2));
  fflush(stdout);  
  printf("Correct Scan(N-1) = %f, Parallel_Scan(N-1) = %f \n", N, y(N-1));
  fflush(stdout);  

  ASSERT_EQ(y(0), 1);
  ASSERT_EQ(y(2), 3);
  ASSERT_EQ(y(N-1), N);

}
*/
//#endif //KR_ENABLE_RESILIENT_EXECUTION_SPACE
//#endif //KR_ENABLE_OPENMP
