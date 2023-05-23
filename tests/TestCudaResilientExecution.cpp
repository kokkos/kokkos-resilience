/*
 */

#include <gtest/gtest.h>
#include "TestCommon.hpp"
#include <Kokkos_Core.hpp>
#include <Kokkos_Core_fwd.hpp>
#include "resilience/Resilience.hpp"
#include "resilience/cuda/ResCudaSpace.hpp"
#include "resilience/cuda/ResCuda.hpp"

#include <thread>
#include <vector>
#include <omp.h>
#include <cstdio>

#define N 20
#define N_2 10
#define MemSpace KokkosResilience::ResCudaSpace
#define ExecSpace KokkosResilience::ResCuda

//Resilient
using range_policy = Kokkos::RangePolicy<ExecSpace>;
using ResVectorDoubleType = Kokkos::View< double*, Kokkos::LayoutLeft, MemSpace,
			      Kokkos::Experimental::SubscribableViewHooks<
				KokkosResilience::CudaResilientSubscriber>>;

//Non-resilient
using ViewVectorType = Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::CudaSpace>;
using range_policy2 = Kokkos::RangePolicy<Kokkos::Cuda>;

static_assert (!Kokkos::Impl::MemorySpaceAccess<Kokkos::HostSpace, MemSpace>::accessible);


// gTest runs CUDA resilient execution space and performs a deep copy to show space
// itself is working.
TEST(CudaTestRig, TestCudaDeepCopy)
{

  std::cout << "Entered TestCudaSpace Test successfully." << std::endl;
  
  // Create device view
  
  range_policy test;
  
  using namespace std::string_literals;

  
  ResVectorDoubleType x (Kokkos::view_alloc ( "x"s, Kokkos::WithoutInitializing), N );
  ResVectorDoubleType y (Kokkos::view_alloc ( "y"s, Kokkos::WithoutInitializing), N );
  // ResVectorDoubleType y( "y", N );

  std::cout << "Created device views" << std::endl;

  // Create host mirror of device view
  auto h_x = Kokkos::create_mirror_view( x );
  auto h_y = Kokkos::create_mirror_view( x );

  Kokkos::fence();

  std::cout << "Got past Host Mirror views" << std::endl; 
  std::printf("h_x1 is %lf\n", h_x(1));
  
  // Initialize x vector on host
  for ( int i = 0; i < N; i++ ) {
    h_x( i ) = i;
  }

  std::cout << "h_x1 is now " << h_x (1) << std::endl;  

  std::cout << "Initialized x host mirror vector h_x = i on host" << std::endl;

  // Deep copy host to device (dest, src), checking if space works
  // Uses ParallelFor code, HostSpace -> ResCudaSpace/passthrough to CudaSpace
  Kokkos::deep_copy( x, h_x );

  // ResCudaSpace/pass to CudaSpace -> ResCudaSpace/pass to CudaSpace
  Kokkos::deep_copy( y, x );

  // ResCudaSpace/pass to CudaSpace -> HostSpace
  Kokkos::deep_copy( h_y, y);

  std::cout << "Got past the deepcopies." << std::endl << std::endl;

  // Assert that the deepcopies took
  for ( int i = 0; i < N; i++) {
    ASSERT_EQ(h_y(i), i);
  }

}

/*********************************
*********PARALLEL FORS************
**********************************/

void test_kokkos_for(){

  using namespace std::string_literals;

  // Create device views
  ResVectorDoubleType x (Kokkos::view_alloc ( "x"s, Kokkos::WithoutInitializing), N );
  ResVectorDoubleType y (Kokkos::view_alloc ( "y"s, Kokkos::WithoutInitializing), N );

  // Create host mirrors of device views.
  auto h_y = Kokkos::create_mirror_view( y );
  auto h_x = Kokkos::create_mirror_view( x );

  Kokkos::fence();
 
  // Initialize y vector on host
  for ( int i = 0; i < N; i++ ){
    h_y( i ) = i;
  }

  // Deep copy host to device (dest, src)
  Kokkos::deep_copy( y, h_y );

  std::cout<< "This print is from right before the parallel_for, where we think everything is erroring. " << std::endl;

  // Copy y to x on device using parallel for
  Kokkos::parallel_for( range_policy(0, N), KOKKOS_LAMBDA(int i) { 
    x (i) = y (i) * y (i); 
  });

  // Deep copy device to host (dest, src)
  Kokkos::deep_copy(h_x, x);
  
  Kokkos::fence();  

  // Assert that the parallel for took
  for ( int i = 0; i < N; i++) {
    ASSERT_EQ(h_x(i), i*i);
  }
  
  KokkosResilience::clear_duplicates_cache();
}

// gTest runs CUDA parallel_for with non-resilient Kokkos. Should never fail.
TEST(CudaTestRig, TestKokkosFor)
{
 
  test_kokkos_for();

}


