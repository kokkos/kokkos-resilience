#include <gtest/gtest.h>
#include "TestCommon.hpp"
#include <Kokkos_Core.hpp>
#include <Kokkos_Core_fwd.hpp>
#include <resilience/Resilience.hpp>
#include <resilience/OpenMP/ResHostSpace.hpp>
#include <resilience/OpenMP/ResOpenMP.hpp>

//#ifdef KR_ENABLE_OPENMP // TODO: RESILIENCE HEADER MODIFY
//#ifdef KR_ENABLE_RESILIENT_EXECUTION_SPACE // TODO: REIMPLEMENT
//!!!! And possibly other macros, check

#define N 5
#define MemSpace KokkosResilience::ResHostSpace
#define ExecSpace KokkosResilience::ResOpenMP

/*TEST(TestResOpenMP, gTestFunctioning)
{
  printf("Arrived in TestResOpenMP, gTestFunctioning\n");
  int x = 1;
  ASSERT_EQ(x, 1);

  printf("\n\n\n");
  fflush(stdout);

}

// gTest if the spaces work by themselves. Goal is to deepcopy one view to another.
TEST(TestResOpenMP, TestSpaces)
{

  using range_policy = Kokkos::RangePolicy<ExecSpace>;

  typedef Kokkos::View<double*, Kokkos::LayoutRight, MemSpace> ViewVectorType;

  ViewVectorType y( "y", N);
  ViewVectorType x( "x", N);

  for ( int i = 0; i < N; i++ ) {
    y( i ) = 1;
  }

  // Time the deepcopy
  Kokkos::Timer timer;
  Kokkos::deep_copy(x, y);
  double time = timer.seconds();

  std::cout << "The as-yet unconfirmed deep-copy took " << time << " seconds." << std::endl;

  for ( int i = 0; i < N; i++) {
    //printf("x[%d]=%f\n", i, x(i));
    ASSERT_EQ(x(i), 1);
  }
    
  printf("\n\n\n");
  fflush(stdout);

}*/

/*********************************
*********PARALLEL FORS************
**********************************/


// gTest if the ParallelFor test works with regular Kokkos.
TEST(TestResOpenMP, TestRegularFor)
{
 
  using range_policy = Kokkos::RangePolicy<Kokkos::OpenMP>;

  // Allocate y, x vectors.
  typedef Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::HostSpace>   ViewVectorType;
  ViewVectorType y( "y", N );
  ViewVectorType x( "x", N );

  // Timer
  Kokkos::Timer timer;

  // Initialize y vector on host using parallel_for
  Kokkos::parallel_for( range_policy (0, N), KOKKOS_LAMBDA ( const int i) {
    y ( i ) = i;
  });
 
  Kokkos::fence();
  
  // Calculate time.
  double time = timer.seconds();
  
  Kokkos::deep_copy(x, y);
  for ( int i = 0; i < N; i++) {
    //printf("x[%d]=%f\n", i, x(i));
    ASSERT_EQ(x(i), i);
  }

  printf("It took %f seconds to perform the parallel_for, not including deep-copy\n", time);

  printf("\n\n\n");
  fflush(stdout);

}
 
// The gtest checking if the for works. Goal is to get into the parallel_for at all.
///*
TEST(TestResOpenMP, TestParallelFor)
{
  printf("Entered the test. This is the first line of code.\n");
  fflush(stdout);

  using range_policy = Kokkos::RangePolicy<ExecSpace>;

  // Allocate y, x vectors.
  typedef Kokkos::View<double*, Kokkos::LayoutRight, MemSpace>   ViewVectorType;
  ViewVectorType y( "y", N );
  ViewVectorType x( "x", N );
  
  printf("Got past declaring the vectors.\n"); 
  fflush(stdout);

  // Timer
  Kokkos::Timer timer;

  printf("Got past the Timer in the test.\n");
  fflush(stdout);
  
  int set_data_counter = 0;

  //Initialize y vector on host using parallel_for
  Kokkos::parallel_for( range_policy (0, N), KOKKOS_LAMBDA ( const int i) {
    y ( i ) = i;
    set_data_counter++; 
  });

  Kokkos::fence(); //Is this needed? Fence in resilient parallel_for
 
  printf("Got past the For in the test.\n");
  fflush(stdout);

  // Calculate time.
  double time = timer.seconds();

  Kokkos::deep_copy(x, y);

  for ( int i = 0; i < N; i++) {
    //printf("x[%d]=%d\n", i, x(i));
    ASSERT_EQ(x(i), i);
  }

  printf("Got past the old test-assert, setting data.\n");
  fflush(stdout);

  printf("Setdatacounter is %d. It should be %d.\n", set_data_counter, N);
  fflush(stdout);

  ASSERT_EQ(set_data_counter, N);

  printf("\n\n\n");
  fflush(stdout);
}

/**********************************
********PARALLEL REDUCES***********
**********************************/
/*
// gTest if the ParallelReduce test works with regular Kokkos.
TEST(TestResOpenMP, TestRegularReduce)
{
 
  using range_policy = Kokkos::RangePolicy<Kokkos::OpenMP>;

  // Allocate y, x vectors.
  typedef Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::HostSpace>   ViewVectorType;
  ViewVectorType y( "y", N );
  ViewVectorType x( "x", N );

  double result = 0;
  double correct = N;

  // Initialize y vector on host using parallel_for
  Kokkos::parallel_for( range_policy (0, N), KOKKOS_LAMBDA ( const int i) {
    y ( i ) = 1;
  });
 
  Kokkos::fence();
  
  Kokkos::deep_copy(x, y);

  // Timer
  Kokkos::Timer timer;

  // Perform vector dot product y*x using parallel_reduce
  Kokkos::parallel_reduce( "yx", N, KOKKOS_LAMBDA ( int j, double &update ) {
    update += y( j ) * x( j );
  }, result );

  Kokkos::fence();

  // Calculate time.
  double time = timer.seconds(); 
 
  printf("It took %f seconds to perform the parallel_for, not including deep-copy\n", time);
  fflush(stdout);
  printf("The correct length of two all ones vectors multiplied together is N. N is %f.\n", correct);
  fflush(stdout);
  printf("The result from parallel_reduce was %f. \n", result);
  fflush(stdout);
  
  printf("\n\n\n");
  fflush(stdout);

  ASSERT_EQ(result, correct);

}*/
/*
// gTest if the Resilient parallel_reduce works.
TEST(TestResOpenMP, TestParallelReduce)
{
 
  using range_policy = Kokkos::RangePolicy<ExecSpace>;

  // Allocate y, x vectors.
  typedef Kokkos::View<double*, Kokkos::LayoutRight, MemSpace> ViewVectorType;
  ViewVectorType y( "y", N );
  ViewVectorType x( "x", N );

  double result = 0;
  double correct = N;

  // Initialize y vector on host using regular for (isolate parallel_reduce)
  for ( int i = 0; i < N; i++ ) {
    y( i ) = 1;
  }
  
  Kokkos::deep_copy(x, y);

  // Timer
  Kokkos::Timer timer;

  // Perform vector dot product y*x using parallel_reduce
  Kokkos::parallel_reduce( "yx", N, KOKKOS_LAMBDA ( int j, double &update ) {
    update += y( j ) * x( j );
  }, result );

  Kokkos::fence();

  // Calculate time.
  double time = timer.seconds(); 
 
  printf("It took %f seconds to perform the parallel_for, not including deep-copy\n", time);
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
    x ( i ) = 1;
  });
 
  Kokkos::fence();
 
  // Timer
  Kokkos::Timer timer;

  // Perform vector element sum using parallel_scan
  Kokkos::parallel_scan( "xscan", N, KOKKOS_LAMBDA ( int j, double &update, const bool& final ) {
    update += x( j );
    printf("Update: %f \n", update);
    fflush(stdout);
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
*/
/*
// gTest if the parallel_scan test works with resilient Kokkos.
TEST(TestResOpenMP, TestParallelScan)
{
 
  using range_policy = Kokkos::RangePolicy<ExecSpace>;

  // Allocate x vector.
  typedef Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::HostSpace>   ViewVectorType;
  ViewVectorType x( "x", N );

  // Initialize x vector on host using regular for (isolate parallel_scan)
  for ( int i = 0; i < N; i++ ) {
    x( i ) = 1;
  }
   
  // Timer
  Kokkos::Timer timer;

  // Perform vector element sum using parallel_scan
  Kokkos::parallel_scan( "xscan", N, KOKKOS_LAMBDA ( int j, double &update, const bool& final ) {
    update += x( j );
    printf("Update: %f \n", update);
    fflush(stdout);
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
  printf("Correct Scan(0) = %f, Parallel_Scan(0) = %f \n", 1, x(0));
  fflush(stdout);
  printf("Correct Scan(2) = %f, Parallel_Scan(2) = %f \n", 3, x(2));
  fflush(stdout);  
  printf("Correct Scan(N-1) = %f, Parallel_Scan(N-1) = %f \n", N, x(N-1));
  fflush(stdout);  

  ASSERT_EQ(x(0), 1);
  ASSERT_EQ(x(2), 3);
  ASSERT_EQ(x(N-1), N);

}*/

//#endif //KR_ENABLE_RESILIENT_EXECUTION_SPACE
//#endif //KR_ENABLE_OPENMP