#include <gtest/gtest.h>
#include "TestCommon.hpp"
#include <Kokkos_Core.hpp>
#include <Kokkos_Core_fwd.hpp>
#include <resilience/Resilience.hpp>
#include <resilience/OpenMP/ResHostSpace.hpp>
#include <resilience/OpenMP/ResOpenMP.hpp>

//#ifdef KR_ENABLE_OPENMP
//#ifdef KR_ENABLE_RESILIENT_EXECUTION_SPACE
//!!!! And possibly other macros, check

#define N 5
#define MemSpace KokkosResilience::ResHostSpace
#define ExecSpace KokkosResilience::ResOpenMP

TEST(TestResOpenMP, gTestFunctioning)
{
  printf("Arrived in TestResOpenMP, gTestFunctioning\n");
  int x = 1;
  ASSERT_EQ(x, 1);
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

  Kokkos::deep_copy(x, y);

  for ( int i = 0; i < N; i++) {
    //printf("x[%]=%f\n", i, x(i));
    ASSERT_EQ(x(i), 1);
  }

}


// The gtest checking if the for works. Goal is to get into the parallel_for at all.
///*

TEST(TestResOpenMP, TestParallelFor)
{

  using range_policy = Kokkos::RangePolicy<ExecSpace>;

  // Allocate y, x vectors.
  typedef Kokkos::View<double*, Kokkos::LayoutRight, MemSpace>   ViewVectorType;
  typedef Kokkos::View<double**, Kokkos::LayoutRight, MemSpace>  ViewMatrixType;
  ViewVectorType y( "y", N );
  ViewVectorType x( "x", N );
    
  // Timer products.
  Kokkos::Timer timer;

  // Initialize y vector on host using parallel_for
  double result = 0;
  Kokkos::parallel_for( range_policy (0, N), KOKKOS_LAMBDA ( const int i) {
    y ( i ) = i;
  });

  Kokkos::fence(); //Is this needed? Fence in resilient parallel_for
 
  // Calculate time.
  double time = timer.seconds();

  Kokkos::deep_copy(x, y);

  for ( int i = 0; i < N; i++) {
    printf("x[%]=%d\n", i, x(i));
    ASSERT_EQ(x(i), i);
  }

}
//*/
//#endif //KR_ENABLE_RESILIENT_EXECUTION_SPACE
//#endif //KR_ENABLE_OPENMP
