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
#include <resilience/exec_space/openMP/Resilient_HostSpace.hpp>
#include <resilience/exec_space/openMP/Resilient_OpenMP.hpp>

#include <thread>
#include <vector>
#include <omp.h>
#include <cstdio>

#define N 1000
#define N_2 500
#define M 100
#define test_const 10
#define MemSpace KokkosResilience::ResHostSpace
#define ExecSpace KokkosResilience::ResOpenMP

//Resilient
using res_range_policy = Kokkos::RangePolicy<ExecSpace>;

template<typename DataType, typename... MemoryTraits>
using ResilientView = Kokkos::View<
	 	      DataType,
		      MemSpace,
		      Kokkos::Experimental::SubscribableViewHooks< 
		              KokkosResilience::ResilientDuplicatesSubscriber>,
		      MemoryTraits...
		      >;

//Non-resilient
using KokkosVectorView = Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::HostSpace>;
using omp_range_policy = Kokkos::RangePolicy<Kokkos::OpenMP>;

/*********************************
*********PARALLEL FORS************
**********************************/

// gTest runs parallel_for with non-resilient Kokkos. Should never fail.
TEST(TestResOpenMP, TestKokkosFor)
{
  std::cout << "N was set at: " << N << std::endl;
  // Allocate y, x vectors.
  KokkosVectorView y2( "y", N );
  KokkosVectorView x2( "x", N );

  // Initialize y vector on host using parallel_for
  Kokkos::parallel_for("GTestParallelDoubleFor",
    omp_range_policy(0, N), KOKKOS_LAMBDA(int i) { y2(i) = i; });

  Kokkos::deep_copy(x2, y2);
  
  for ( int i = 0; i < N; i++) {
    ASSERT_EQ(x2(i), i);
  }
}

// gTest runs parallel_for with resilient Kokkos doubles assignment and atomic counter.
// Expect counter to count iterations.
TEST(TestResOpenMP, TestResilientForDouble)
{
  
  KokkosResilience::global_error_settings = KokkosResilience::Error(0.001);
  
  // Allocate y, x vectors.
  ResilientView<double*> y( "y", N );
  ResilientView<double*> x( "x", N );

  //Integer vector 1 long to count data accesses, because scalar view bugs (previously)
  ResilientView<int*> counter( "DataAccesses", 1);

  counter(0) = 0;

  //Initialize y vector on host using parallel_for, increment a counter for data accesses.
  Kokkos::parallel_for("GTestResilientDoubleFor", res_range_policy (0, N), KOKKOS_LAMBDA ( const int i) {
    y ( i ) = i;
    Kokkos::atomic_inc(&counter(0));
  });

  //reset global error settings
  KokkosResilience::ErrorInject::error_counter=0;
  KokkosResilience::global_error_settings.reset();
  KokkosResilience::print_total_error_time();
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
  ResilientView<int*> y( "y", N );
  ResilientView<int*> x( "x", N );

  //Integer vector 1 long to count data accesses, because scalar view bugs (previously)
  ResilientView<int*>  counter( "DataAccesses", 1);

  counter(0) = 0;

  //Initialize y vector on host using parallel_for, increment a counter for data accesses.
  Kokkos::parallel_for( res_range_policy (0, N), KOKKOS_LAMBDA ( const int i) {
    y ( i ) = i;
    Kokkos::atomic_inc(&counter(0));
  });
  
  KokkosResilience::clear_duplicates_cache();
  
  Kokkos::deep_copy(x, y);
  
  for ( int i = 0; i < N; i++) {
    ASSERT_EQ(x(i), i);
  }
  ASSERT_EQ(counter(0), N);
}

// Test handler for unrecoverable data corruption
TEST(TestResOpenMP, TestErrorHandler)
{

  ResilientView<int*> counter ( "DataAccesses", 1);

  // Allocate y, x vectors.
  ResilientView<double*> y( "y", N );
  ResilientView<double*> x( "x", N );

  counter(0) = 0;

  bool failed_recovery = false;
  KokkosResilience::set_unrecoverable_data_corruption_handler(
      [&failed_recovery](std::size_t) { failed_recovery = true; });
 
  KokkosResilience::ErrorInject::error_counter = 0;
  //Set an extremely high error rate that the test cannot recover from
  //Half of all values are errors
  KokkosResilience::global_error_settings = KokkosResilience::Error(0.5);
  
  Kokkos::parallel_for( res_range_policy (0, N), KOKKOS_LAMBDA ( int i) {
    y(i) = counter(0);
    Kokkos::atomic_inc(&counter(0));
  });

  KokkosResilience::global_error_settings.reset();
  KokkosResilience::print_total_error_time();
  KokkosResilience::clear_duplicates_cache();

  KokkosResilience::set_unrecoverable_data_corruption_handler(&KokkosResilience::default_unrecoverable_data_corruption_handler);

  ASSERT_TRUE(failed_recovery);
}

// gTest runs parallel_for with resilient Kokkos doubles assignment and atomic counter.
// Uses a non-zero range start
TEST(TestResOpenMP, TestResilientNonZeroRange)
{

  // Allocate y, x vectors.
  ResilientView<double*> y( "y", N );
  ResilientView<double*> x( "x", N );

  //Initialize y vector on host using parallel_for, increment a counter for data accesses.
  Kokkos::parallel_for( res_range_policy (0, N_2), KOKKOS_LAMBDA ( const int i) {
    y ( i ) = 1;
  });

  Kokkos::parallel_for( res_range_policy (N_2, N), KOKKOS_LAMBDA ( const int i) {
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

  ResilientView<double*> x( "x", N );
  ResilientView<double*> y( "y", N );

  Kokkos::parallel_for( res_range_policy (0, N), KOKKOS_LAMBDA ( const int i) {
    x ( i ) = i;
  });

  ResilientView<const double*> x_const = x;

  Kokkos::parallel_for( res_range_policy (0, N), KOKKOS_LAMBDA ( const int i) {
    y ( i ) = 2 * x_const (i);
  });

  KokkosResilience::clear_duplicates_cache();

  for ( int i = 0; i < N; i++) {
    ASSERT_EQ(y(i), 2 * i);
  }

}

// KOKKOS MULTIDIMENSIONAL TEST
// gTest runs parallel_for with Kokkos doubles assignment and atomic counter,
// on a multidimensional view.
// Expect counter to count iterations.
TEST(TestResOpenMP, TestKokkos2D)
{
// Allocate 2D y, x vectors.
  Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> x( "x", N, N );
  Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> y( "y", N, N );

  //Kokkos::View<int, Kokkos::HostSpace> counter;  
  Kokkos::View<int*, Kokkos::LayoutRight, Kokkos::HostSpace> counter( "DataAccesses", 1);  

  counter(0) = 0;
 
  //Initialize y vector on host using parallel_for, increment a counter for data accesses.
  Kokkos::parallel_for( omp_range_policy (0, N), KOKKOS_LAMBDA ( const int i) {
    for (int j = 0; j < N; j++){
      y ( i,j ) = i+j;
      Kokkos::atomic_inc(&counter(0));
    }
  });

  Kokkos::deep_copy(x,y);
  
  for ( int i = 0; i < N; i++) {
    for ( int j = 0; j < N; j++) {
      ASSERT_EQ(x(i,j), i+j);
    }
  }
  ASSERT_EQ(counter(0), N*N);
 
  std::cout << std::endl <<std::endl;
 
}

// gTest runs parallel_for with resilient Kokkos doubles assignment
// and atomic counter on a multidimensional view.
// Expect counter to count accesses to each vector element.
TEST(TestResOpenMP, TestResilient2D)
{

  KokkosResilience::ErrorInject::error_counter = 0;
  KokkosResilience::global_error_settings = KokkosResilience::Error(0.0001);
	
  // Allocate y, x vectors.
  ResilientView<double**> y( "y", N, N );
  ResilientView<double**> x( "x", N, N );

 
  //Initialize y vector on host using parallel_for, increment a counter for data accesses.
  Kokkos::parallel_for( res_range_policy (0, N), KOKKOS_LAMBDA ( const int i) {
    for (int j = 0; j < N; j++){
      y ( i,j ) = i+j;
    }
  });

  KokkosResilience::print_total_error_time();
  KokkosResilience::clear_duplicates_cache(); 
  KokkosResilience::ErrorInject::error_counter = 0;
  KokkosResilience::global_error_settings.reset();

  Kokkos::deep_copy(x, y);

  for ( int i = 0; i < N; i++) {
    for ( int j = 0; j < N; j++) {
      ASSERT_EQ(x(i,j), i+j);
    }
  }
}

// gTest runs parallel_for with Kokkos doubles assignment and atomic counter,
// on a multidimensional view *[N]
// Expect counter to count iterations.
TEST(TestResOpenMP, TestKokkos2DPad)
{

  //  KokkosResilience::global_error_settings = KokkosResilience::Error(0.01);
  ResilientView<double*[N]> x( "x", N );
  ResilientView<double*[N]> y( "y", N );
  ResilientView<int*, Kokkos::MemoryTraits <Kokkos::Atomic> > counter( "counter", 1);

  size_t rank = x.rank();
  std::cout << "The rank of View x is rank: " << rank << "\n";

  counter(0) = 0;

  //Initialize y vector on host using parallel_for, increment a counter for data accesses.
  Kokkos::parallel_for( res_range_policy (0, N), KOKKOS_LAMBDA ( const int i) {
    for (int j = 0; j < N; j++){
      y ( i,j ) = i+j;
      counter(0)++;
    }
  });

  KokkosResilience::clear_duplicates_cache();
  //KokkosResilience::print_total_error_time();
  //KokkosResilience::global_error_settings.reset();

  Kokkos::deep_copy(x,y);

  for ( int i = 0; i < N; i++) {
    for ( int j = 0; j < N; j++) {
      ASSERT_EQ(x(i,j), i+j);
    }
  }
  ASSERT_EQ(counter(0), N*N);
}

//Test MiniMD Exact Kernel Behavior
//Test worked, then was rewritten to demonstrate failure due to read/write copy error
//Copy-in-place and overwrite original with result leads to massive error
TEST(TestResOpenMP, TestMiniMDKernel)
{

  // Allocate 2D y, x vectors.
  Kokkos::View<double*[2],Kokkos::LayoutRight, Kokkos::HostSpace > x( "x", M );
  Kokkos::View<double*[2],Kokkos::LayoutRight, Kokkos::HostSpace > y( "y", M );
  Kokkos::View<double*[2],Kokkos::LayoutRight, Kokkos::HostSpace > z( "z", M );

  //Initialize x vector REGULAR kernel
  Kokkos::parallel_for( omp_range_policy (0, M), KOKKOS_LAMBDA ( const int i) {
     x ( i,0 ) = 1;
  });

  int j = 0;

  while (j<5){
    //Test MiniMD Kernel Behavior with RESILIENT kernel, NONRESILEINT views
    Kokkos::parallel_for( res_range_policy (0, M), KOKKOS_LAMBDA ( const int i) {
      y ( i, 0 ) += test_const * x ( i, 0 );
      z ( i, 0 ) += test_const * y ( i, 0 );
    });
    j++;
  }

  KokkosResilience::clear_duplicates_cache();

  std::cout << "Test values y(1,0) and z(1,0) are " << y(1,0) << " and " << z(1,0) << " respectively\n. ";
  std::cout << "This test should have resulted in 50 and 1500, in these integrations, respectively." << std::endl;

  for ( int i = 0; i < M; i++) {
    ASSERT_EQ(y(i,0), 15*test_const );
    ASSERT_EQ(z(i,0), 120*test_const*test_const );
  }
}

//Test MiniMD Exact Kernel Behavior with Resilience
TEST(TestResOpenMP, TestMiniMDKernelResilient)
{
  KokkosResilience::ErrorInject::error_counter = 0;
  std::cout << "ErrorInject::error_counter is " << KokkosResilience::ErrorInject::error_counter << "\n";
  std::cout << "This is the test of minMD 2D Resilient Error Injection \n\n\n";
  KokkosResilience::global_error_settings = KokkosResilience::Error(0.001);

  // Allocate 2D y, x vectors.
  ResilientView<double*[2]> x( "x", M );
  ResilientView<double*[2]> y( "y", M );
  ResilientView<double*[2]> z( "z", M );

  //Initialize x vector RESIIENT kernel WITH ERRORS
  Kokkos::parallel_for( res_range_policy (0, M), KOKKOS_LAMBDA ( const int i) {
     x ( i,0 ) = 1;
  });

  int j = 0;

  while (j<5){
    //Test MiniMD Kernel Behavior with RESILIENT kernel, RESILEINT views WITH ERRORS (cont prev count)
    Kokkos::parallel_for( res_range_policy (0, M), KOKKOS_LAMBDA ( const int i) {
      y ( i, 0 ) += test_const * x ( i, 0 );
      z ( i, 0 ) += test_const * y ( i, 0 );
    });
    j++;
  }

  for ( int i = 0; i < M; i++) {
    ASSERT_EQ(y(i,0), 5*test_const );
    ASSERT_EQ(z(i,0), 15*test_const*test_const );
  }

  KokkosResilience::print_total_error_time();
  KokkosResilience::clear_duplicates_cache();
  KokkosResilience::ErrorInject::error_counter=0;
  KokkosResilience::global_error_settings.reset();

  std::cout << std::endl <<std::endl;

}

//Test RandomAccess
TEST(TestResOpenMP, TestRandomAccess)
{
  // Allocate RandomAccess Subscribers x, y vectors.
  ResilientView<double*, Kokkos::MemoryTraits <Kokkos::RandomAccess> > x( "x", N );
  ResilientView<double*, Kokkos::MemoryTraits <Kokkos::RandomAccess> > y( "y", N );

  Kokkos::parallel_for( res_range_policy (0, N), KOKKOS_LAMBDA ( const int i) {
    y ( i ) = i;
  });

  KokkosResilience::clear_duplicates_cache();

  Kokkos::deep_copy(x, y);
  for ( int i = 0; i < N; i++) {
    ASSERT_EQ(x(i), i);
  }
}

// Test Atomic (As Memory Trait)
TEST(TestResOpenMP, TestAtomic)
{
  // Allocate Atomic Subscribers x, y vectors.
  // See if counter behaves as if counter was accessed atomically.
  ResilientView<double*, Kokkos::MemoryTraits <Kokkos::Atomic> > x( "x", N );
  ResilientView<double*, Kokkos::MemoryTraits <Kokkos::Atomic> > y( "y", N );
  ResilientView<int*, Kokkos::MemoryTraits <Kokkos::Atomic> > counter( "counter1", 1 );
  Kokkos::View<int*, Kokkos::LayoutRight, Kokkos::HostSpace> counter2( "counter2", 1);

  Kokkos::parallel_for( res_range_policy (0, N), KOKKOS_LAMBDA ( const int i) {
    y ( i ) = i;
    counter(0)++;
    Kokkos::atomic_inc(&counter2(0));
  });

  std::cout << "Counter (resilient, atomic declared) is: " << counter(0) << std::endl;
  std::cout << "Counter2 (non-resilient, atomic access) is: " << counter2(0) << std::endl;
  std::cout << std::endl;

  KokkosResilience::clear_duplicates_cache();

  Kokkos::deep_copy(x, y);
  for ( int i = 0; i < N; i++) {
    ASSERT_EQ(x(i), i);
  }
}
