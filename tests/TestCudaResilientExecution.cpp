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
#include <resilience/Resilience.hpp>
#include <resilience/exec_space/cuda/Resilient_CudaSpace.hpp>
#include <resilience/exec_space/cuda/Resilient_Cuda.hpp>

#include <thread>
#include <vector>
#include <cstdio>
#include <string>

#define N 100
#define N_2 50
#define M 100
#define test_const 10
#define MemSpace KokkosResilience::ResCudaSpace
#define ExecSpace KokkosResilience::ResCuda

//Resilient
using res_range_policy = Kokkos::RangePolicy<ExecSpace>;


template<typename DataType, typename... MemoryTraits>
using ResilientView = Kokkos::View<
                      DataType,
		      Kokkos::LayoutLeft,
                      MemSpace,
                      Kokkos::Experimental::SubscribableViewHooks<
                              KokkosResilience::ResilientDuplicatesSubscriber>,
                      MemoryTraits...
                      >;


//Non-resilient
template<typename DataType, typename... MemoryTraits>
using KokkosCudaView = Kokkos::View<
                       DataType,
                       Kokkos::LayoutLeft,
                       Kokkos::CudaSpace,
                       MemoryTraits...
                       >;
using cuda_range_policy = Kokkos::RangePolicy<Kokkos::Cuda>;

TEST(TestResCuda, TestCudaSpace)
{

  // Create device view
  KokkosCudaView<double*> x( "x", N );
  KokkosCudaView<double*> y( "y", N );

  // Create host mirror of device view
  auto h_x = Kokkos::create_mirror( x );
  auto h_y = Kokkos::create_mirror( y );

  static_assert( std::same_as<decltype(h_x)::memory_space, Kokkos::HostSpace > );

  // Initialize x vector on host
  for ( int i = 0; i < N; i++ ){
    h_x( i ) = i;
  }

  // Deep copy device to host (dest, src), checking if space works
  // Using deep-copy only, should pass information host->device->device->host
  Kokkos::deep_copy( x, h_x );

  Kokkos::deep_copy ( y, x);
  Kokkos::deep_copy ( h_y, y);

  for (int i=0; i< N; i++) {
    ASSERT_EQ( h_y(i), i);
  }

  Kokkos::fence();

}

// Tests if ResCudaSpace works and can pass information between execution space types 
TEST(TestResCuda, TestResCudaSpace)
{
  
  // Create device view - without-initializing is not mandatory
  ResilientView<double*> x( Kokkos::view_alloc("x", Kokkos::WithoutInitializing), N );
  ResilientView<double*> y( Kokkos::view_alloc("y", Kokkos::WithoutInitializing), N );
  
  // Create host mirror of device view
  auto h_x = Kokkos::create_mirror( x );
  auto h_y = Kokkos::create_mirror( y );

  // Initialize x vector on host
  for ( int i = 0; i < N; i++ ){
    h_x( i ) = i;
  }

  // Deep copy device to host (dest, src), checking if space works
  // Using deep-copy only, should pass information host->device->device->host
  Kokkos::deep_copy( x, h_x );

  Kokkos::deep_copy ( y, x);
  Kokkos::deep_copy ( h_y, y);

  for (int i=0; i< N; i++) {
    ASSERT_EQ( h_y(i), i);
  }

  Kokkos::fence();
  KokkosResilience::clear_duplicates_cache();

}
//#endif
/*********************************
*********PARALLEL FORS************
**********************************/

// Function for gtest testing Kokkos parallel fors, should always work
void test_kokkos_for(){

  // Create device views
  KokkosCudaView<double*> y( "y", N );
  KokkosCudaView<double*> x( "x", N );

  // Create host mirrors of device views.
  auto h_y = Kokkos::create_mirror_view( y );
  auto h_x = Kokkos::create_mirror_view( x );

  // Initialize y vector on host
  for ( int i = 0; i < N; i++ ){
    h_y( i ) = i;
  }

  // Deep copy host to device (dest, src)
  Kokkos::deep_copy( y, h_y );

  // A trivial parallel_for multiplication
  Kokkos::parallel_for(cuda_range_policy(0, N), KOKKOS_LAMBDA(int i) { 
	  x (i) = y (i) * y (i); 
	  });

  // Deep copy device to host (dest, src)
  Kokkos::deep_copy(h_x, x);
 
  for ( int i = 0; i < N; i++) {
    ASSERT_EQ(h_x(i), i*i);
  }

}

// CUDA parallel_for with non-resilient Kokkos. Should never fail.
TEST(TestResCuda, TestKokkosFor)
{
 
  test_kokkos_for();

}

// function for gtest CUDA parallel_for with resilient Kokkos using doubles in view entries
void test_res_for(){

  ResilientView<double*> x( "x", N );
  ResilientView<double*> y( "y", N );
#if defined KR_ERROR_INJECTION
  KokkosResilience::ErrorInjectionTracking::error_counter = 0;
  KokkosResilience::global_error_settings = KokkosResilience::Error(0.0001);
#endif
  // Create host mirror of device view
  auto h_x = Kokkos::create_mirror( x );
  auto h_y = Kokkos::create_mirror( y );

  // Initialize y vector on host
  for ( int i = 0; i < N; i++ ){
    h_y( i ) = i;
  }

  // Deep copy host to device (dest, src)
  Kokkos::deep_copy( y, h_y );

  // A trivial parallel_for multiplication
  Kokkos::parallel_for(res_range_policy(0, N), KOKKOS_LAMBDA(int i) {
	  x (i) = y (i) * y (i);
          });
  
  // Deep copy device to host (dest, src)
  Kokkos::deep_copy(h_x, x);

  // Error mitigation testing machinery and cache reset
#if defined KR_ERROR_INJECTION  
  KokkosResilience::global_error_settings.reset();
  KokkosResilience::print_total_error_time();
  KokkosResilience::ErrorInjectionTracking::error_counter=0;
  KokkosResilience::ErrorInjectionTracking::elapsed_seconds = {};
  KokkosResilience::ErrorInjectionTracking::total_error_time = {};
#endif  
  KokkosResilience::clear_duplicates_cache();

  // Assert that resilient parallel_for caught and corrected errors
  for ( int i = 0; i < N; i++) {
    ASSERT_EQ(h_x(i), i*i);
  }
}

// CUDA parallel_for with resilient Kokkos using doubles in view entries
TEST(TestResCuda, TestResFor)
{

  test_res_for();	

}

// function for gtest CUDA parallel_for with resilient Kokkos using integers in view entries
void test_res_int_for(){

#if defined KR_ERROR_INJECTION	
  KokkosResilience::global_error_settings = KokkosResilience::Error(0.001);
#endif
  // Allocate y, x vectors.
  ResilientView<int*> y( "y", N );
  ResilientView<int*> x( "x", N );

  //Integer vector 1 long to count data accesses
  ResilientView<int*>  d_counter( "DataAccesses", 1);

  // Create host mirror of device view
  auto h_counter = Kokkos::create_mirror( d_counter );
  auto h_x = Kokkos::create_mirror( x );
  auto h_y = Kokkos::create_mirror( y );

  // Initialize y vector on host
  for ( int i = 0; i < N; i++ ){
    h_y( i ) = i;
  }
  h_counter(0) = 0;

  // Deep copy host to device (dest, src)
  Kokkos::deep_copy( y, h_y );
  Kokkos::deep_copy(d_counter,h_counter);
  
  // A trivial parallel_for multiplication with added atomic access test
  Kokkos::parallel_for(res_range_policy(0, N), KOKKOS_LAMBDA(int i) {
          x (i) = y (i) * y (i);
          Kokkos::atomic_inc(&d_counter(0));
          });

  // Deep copy device to host (dest, src)
  Kokkos::deep_copy(h_x, x);
  Kokkos::deep_copy(h_counter, d_counter);

#if defined KR_ERROR_INJECTION
  // Error mitigation testing machinery and cache reset
  KokkosResilience::print_total_error_time();
  KokkosResilience::ErrorInjectionTracking::error_counter=0;
  KokkosResilience::ErrorInjectionTracking::elapsed_seconds = {};
  KokkosResilience::ErrorInjectionTracking::total_error_time = {};
  KokkosResilience::global_error_settings.reset();
#endif
  KokkosResilience::clear_duplicates_cache();
 
  // Assert that resilient parallel_for caught and corrected errors
  for ( int i = 0; i < N; i++) {
    ASSERT_EQ(h_x(i), i*i);
  }
  ASSERT_EQ(h_counter(0), N);
}

// CUDA parallel_for with resilient Kokkos using integers in view entries
// Expect counter to count iterations
TEST(TestResCuda, TestResilientForInteger)
{

  test_res_int_for();

}

// function for parallel_for gtest with nonzero range 
void test_res_nonzero_range(){

  ResilientView<double*> x( "x", N );
  ResilientView<double*> y( "y", N );

#if defined KR_ERROR_INJECTION
  KokkosResilience::global_error_settings = KokkosResilience::Error(0.001);
#endif

  // Create host mirror of device view
  auto h_x = Kokkos::create_mirror( x );
  auto h_y = Kokkos::create_mirror( y );

  // Initialize y vector on host
  for ( int i = 0; i < N/2; i++ ){
    h_y( i ) = 1.0;
  }
  for ( int i = N/2; i < N; i++ ){
    h_y( i ) = i;
  }

  // Deep copy host to device (dest, src)
  Kokkos::deep_copy( y, h_y );

  // A trivial parallel_for multiplication
  Kokkos::parallel_for(res_range_policy(0, N), KOKKOS_LAMBDA(int i) {
          x (i) = y (i) * y (i);
          });

  // Deep copy device to host (dest, src)
  Kokkos::deep_copy(h_x, x);

#if defined KR_ERROR_INJECTION  
  // Error mitigation testing machinery and cache reset
  KokkosResilience::global_error_settings.reset();
  KokkosResilience::print_total_error_time();
  KokkosResilience::ErrorInjectionTracking::error_counter=0;
#endif  
  KokkosResilience::clear_duplicates_cache();

  // Assert that resilient parallel_for caught and corrected errors
  for ( int i = 0; i < N; i++) {
    if (i<N_2) {
      ASSERT_EQ(h_x(i), 1);
    } else {
      ASSERT_EQ (h_x(i), i*i);
    }
  }
}

// CUDA parallel_for with resilient Kokkos using nonzero range
TEST(TestResCuda, TestResilientNonZeroRange)
{
  test_res_nonzero_range();
}

// function for parallel_for gtest with nonzero range 
void test_res_const_view(){

  ResilientView<double*> x( "x", N );
  ResilientView<double*> y( "y", N );

#if defined KR_ERROR_INJECTION  
  //A higher error rate should be selected to make sure errors inserted in const view
  KokkosResilience::global_error_settings = KokkosResilience::Error(0.01);
#endif

  // Create host mirror of device view
  auto h_x = Kokkos::create_mirror( x );
  auto h_y = Kokkos::create_mirror( y );

  // Initialize y vector on host
  for ( int i = 0; i < N; i++ ){
    h_y( i ) = i;
  }

  // Deep copy host to device (dest, src)
  Kokkos::deep_copy( y, h_y );

  // A trivial parallel_for multiplication
  Kokkos::parallel_for(res_range_policy(0, N), KOKKOS_LAMBDA(int i) {
          x (i) = y (i) * y (i);
          });

  //Create const view
  ResilientView<const double*> x_const = x;

  // Another trivial parallel_for multiplication to trigger errors in const
  Kokkos::parallel_for(res_range_policy(0, N), KOKKOS_LAMBDA(int i) {
          y (i) = 2 * x_const (i);
          });

  // Deep copy device to host (dest, src)
  Kokkos::deep_copy(h_x, y);

#if defined KR_ERROR_INJECTION
  // Error mitigation testing machinery and cache reset
  KokkosResilience::global_error_settings.reset();
  KokkosResilience::print_total_error_time();
  KokkosResilience::ErrorInjectionTracking::error_counter=0;
  KokkosResilience::ErrorInjectionTracking::elapsed_seconds = {};
  KokkosResilience::ErrorInjectionTracking::total_error_time = {};
#endif
  KokkosResilience::clear_duplicates_cache();

  // Assert that resilient parallel_for caught and corrected errors
  for ( int i = 0; i < N; i++) {
    ASSERT_EQ(h_x(i), i*i*2);
  }

}

// Test runs parallel_for with a const view. Expect const view to trigger const view subscriber, a no-op
// Expect non-const view to trigger copies and majority voting resiliency subscriber
TEST(TestResCuda, TestConstViewSubscriber)
{
  test_res_const_view();
}

// gTest runs parallel_for with resilient Kokkos doubles assignment
// and atomic counter on a multidimensional view.
// Expect counter to count accesses to each vector element.
void test_res_2d_view(){

#if defined KR_ERROR_INJECTION	
  KokkosResilience::global_error_settings = KokkosResilience::Error(0.001);
#endif

  // Allocate y, x vectors.
  ResilientView<double**> y( "y", N, N );
  ResilientView<double**> x( "x", N, N );
  ResilientView<int*, Kokkos::MemoryTraits <Kokkos::Atomic> > d_counter( "DataAccesses", 1);

  // Create host mirror of device view
  auto h_x = Kokkos::create_mirror( x );
  auto h_y = Kokkos::create_mirror( y );
  auto h_counter = Kokkos::create_mirror( d_counter );
  int host_counting = 0;

  //Initialize y vector on host using for loops, increment a counter for data accesses
  for ( int i = 0; i < N; i++) {  
    for (int j = 0; j < N; j++){
      h_y ( i,j ) = i+j;
      host_counting++;
    }
  }
  

  //Deep copy host to device (dest, src)
  Kokkos::deep_copy(y, h_y );
  Kokkos::deep_copy( d_counter , h_counter );

  // A trivial parallel_for multiplication to get multi-d data device-device
  Kokkos::parallel_for(res_range_policy(0, N), KOKKOS_LAMBDA(const int i) {
          for (int j = 0; j < N; j++){
	      x (i,j) = 3 * y (i,j);
	      d_counter(0) += 2;
	    }
          });
  // Deep copy device to host (dest, src)
  Kokkos::deep_copy(Kokkos::Cuda(), h_x, x );
  Kokkos::deep_copy(Kokkos::Cuda(), h_counter , d_counter );
  h_counter(0) += host_counting;

#if defined KR_ERROR_INJECTION  
  KokkosResilience::print_total_error_time();
  KokkosResilience::ErrorInjectionTracking::error_counter = 0;
  KokkosResilience::ErrorInjectionTracking::elapsed_seconds = {};
  KokkosResilience::ErrorInjectionTracking::total_error_time = {};
  KokkosResilience::global_error_settings.reset();
#endif
  KokkosResilience::clear_duplicates_cache();

  for ( int i = 0; i < N; i++) {
    for ( int j = 0; j < N; j++) {
      ASSERT_EQ(h_x(i,j), 3*(i+j));
    }
  }
  ASSERT_EQ(h_counter(0), 3*N*N);
}

// gTest runs parallel_for with resilient Kokkos doubles assignment
// and atomic counter on a multidimensional view
// Expect counter to count accesses to each vector element
TEST(TestResCuda, TestResilient2D)
{
  test_res_2d_view();
}

// gTest runs parallel_reduce with resilient Kokkos to get a dot product.
void test_res_reduce(){

#if defined KR_ERROR_INJECTION	
  KokkosResilience::global_error_settings = KokkosResilience::Error(0.01);
#endif     

  // Allocate y, x vectors.
  ResilientView<double*> y( "y", N );
  ResilientView<double*> x( "x", N );

  // Create host mirror of device view
  auto h_x = Kokkos::create_mirror( x );
  auto h_y = Kokkos::create_mirror( y );

  // Initialize y vector on host
  for ( int i = 0; i < N; i++ ){
    h_y( i ) = 1.0;
  }

  // Deep copy host to device (dest, src)
  Kokkos::deep_copy( y, h_y );  
  Kokkos::deep_copy ( x , y );

  double dot_product = 0;

  Kokkos::parallel_reduce(res_range_policy(0,N),KOKKOS_LAMBDA (const int i, double & update) {
    update += x ( i ) * y ( i );
  }, dot_product);

#if defined KR_ERROR_INJECTION
  KokkosResilience::print_total_error_time();
  KokkosResilience::ErrorInjectionTracking::error_counter = 0;
  KokkosResilience::ErrorInjectionTracking::elapsed_seconds = {};
  KokkosResilience::ErrorInjectionTracking::total_error_time = {};
  KokkosResilience::global_error_settings.reset();
#endif
  KokkosResilience::clear_duplicates_cache();

  //Delete, test of combiner stuff that will be deleted after MiniMD
  std::chrono::duration<double> secs = KokkosResilience::combiner_seconds;
  std::cout << "The reduce combiner took " << secs.count() << " seconds" <<std::endl;


  ASSERT_EQ(dot_product, N);

}

TEST(TestResCuda, TestResilientReduce)
{
  test_res_reduce();
}

