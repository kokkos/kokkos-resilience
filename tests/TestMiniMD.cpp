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

#define test_const 10
#define N 100
#define N_2 50
#define MemSpace KokkosResilience::ResHostSpace
#define ExecSpace KokkosResilience::ResOpenMP

//Resilient
using range_policy = Kokkos::RangePolicy<ExecSpace>;
using MD_range_policy = Kokkos::MDRangePolicy<ExecSpace>;
using ViewVectorIntSubscriber = Kokkos::View< int* , Kokkos::LayoutRight, MemSpace,
        Kokkos::Experimental::SubscribableViewHooks<
                KokkosResilience::ResilientDuplicatesSubscriber > >;
using ViewVectorDoubleSubscriber = Kokkos::View< double* , Kokkos::LayoutRight, MemSpace,
        Kokkos::Experimental::SubscribableViewHooks<
                KokkosResilience::ResilientDuplicatesSubscriber > >;
using ViewVectorDoubleSubscriber2D = Kokkos::View< double** , Kokkos::LayoutRight, MemSpace,
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

// gTest runs parallel_for with resilient Kokkos doubles assignment and atomic counter.
// Expect counter to count iterations.
TEST(TestResOpenMP, TestReiterateDoubleP4)
{

  //KokkosResilience::global_error_settings = KokkosResilience::Error(0.1);
	
  // Allocate y, x vectors.
  ViewVectorDoubleSubscriber y( "y", N );
  ViewVectorDoubleSubscriber x( "x", N );

  //Integer vector 1 long to count data accesses, because scalar view bugs (previously)
  ViewVectorIntSubscriber counter( "DataAccesses", 1);

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
  //KokkosResilience::print_total_error_time();
  //KokkosResilience::global_error_settings.reset();
}

// gTest runs parallel_for with Kokkos doubles assignment and atomic counter,
// on a view double **.
// Expect counter to count iterations for all three executions because it is
// not declared resilient.
TEST(TestResOpenMP, TestReiterateKokkos2D)
{

  //KokkosResilience::global_error_settings = KokkosResilience::Error(0.01);

  // Allocate 2D y, x vectors.
  ViewVectorDoubleSubscriber2D x( "x", N, N);
  ViewVectorDoubleSubscriber2D y( "y", N, N);

  Kokkos::View<int*, Kokkos::LayoutRight, Kokkos::HostSpace> counter( "DataAccesses", 1);  

  counter(0) = 0;
 
  //Initialize y vector on host using parallel_for, increment a counter for data accesses.
  Kokkos::parallel_for( range_policy (0, N), KOKKOS_LAMBDA ( const int i) {
    for (int j = 0; j < N; j++){
      y ( i,j ) = i+j;
      Kokkos::atomic_increment(&counter(0));
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
  ASSERT_EQ(counter(0), 3*N*N);
}

// gTest runs parallel_for with Kokkos doubles assignment and atomic counter,
// on a multidimensional view *[N]
// Expect counter to count iterations.
TEST(TestResOpenMP, TestKokkos2DPad)
{

  //  KokkosResilience::global_error_settings = KokkosResilience::Error(0.01);

  // Allocate 2D y, x vectors.
  Kokkos::View<double*[N],Kokkos::LayoutRight, MemSpace,
               Kokkos::Experimental::SubscribableViewHooks<
               KokkosResilience::ResilientDuplicatesSubscriber >> x( "x", N );

  Kokkos::View<double*[N],Kokkos::LayoutRight, MemSpace,
               Kokkos::Experimental::SubscribableViewHooks<
               KokkosResilience::ResilientDuplicatesSubscriber >> y( "y", N );

  Kokkos::View<int*, Kokkos::LayoutRight, MemSpace,
               Kokkos::Experimental::SubscribableViewHooks<
               KokkosResilience::ResilientDuplicatesSubscriber >,
               Kokkos::MemoryTraits <Kokkos::Atomic> > counter( "counter", 1);

  size_t rank = x.rank();
  std::cout << "The rank of View x is rank: " << rank << "\n";

  counter(0) = 0;

  //Initialize y vector on host using parallel_for, increment a counter for data accesses.
  Kokkos::parallel_for( range_policy (0, N), KOKKOS_LAMBDA ( const int i) {
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

// Test what happens with one view not resilient
// TEST DEFECTIVE, only overwrites copies and does not read from them
// Test would fail and not pass if reading as well as writing due to in-place overwriting
TEST(TestResOpenMP, TestOneNonResilientView)
{
 
  ViewVectorDoubleSubscriber y( "y", N );
  ViewVectorType x( "x", N );
  ViewVectorType z( "z", N );
 
  Kokkos::View<int*, Kokkos::LayoutRight, Kokkos::HostSpace> counter( "DataAccesses", 1);

  counter(0) = 0;

  Kokkos::parallel_for( range_policy (0, N), KOKKOS_LAMBDA ( const int i) {
    y ( i ) = i;
    Kokkos::atomic_increment(&counter(0));
  });

  Kokkos::parallel_for( range_policy (0, N), KOKKOS_LAMBDA (const int i) {
    z( i ) = y( i );
    Kokkos::atomic_increment(&counter(0));
  });
 
  KokkosResilience::clear_duplicates_cache();

  Kokkos::deep_copy(x, z);
  for ( int i = 0; i < N; i++) {
    ASSERT_EQ(x(i), i);
  }
}

// All views non-resilient, space only resilient. See what happens.
// Expect failure, but test written to compensate for failed maths
// Counter is counting 3x loops due to non-resilient nature
// If other views read as well as wrote, they would also fail in expected ways
TEST(TestResOpenMP, TestSpaceOnly)
{
  std::cout << "\n\n";

  //KokkosResilience::global_error_settings = KokkosResilience::Error(0.01);

  // Allocate 2D y, x vectors.
  Kokkos::View<double*[N],Kokkos::LayoutRight, Kokkos::HostSpace > x( "x", N );
  Kokkos::View<double*[N],Kokkos::LayoutRight, Kokkos::HostSpace > y( "y", N );

  Kokkos::View<int*, Kokkos::LayoutRight, Kokkos::HostSpace,
               Kokkos::MemoryTraits <Kokkos::Atomic> > counter( "counter", 1);

  size_t rank = x.rank();
  std::cout << "The rank of View x is rank: " << rank << "\n";

  counter(0) = 0;

  //Initialize y vector on host using parallel_for, increment a counter for data accesses.
  Kokkos::parallel_for( range_policy (0, N), KOKKOS_LAMBDA ( const int i) {
    for (int j = 0; j < N; j++){
      y ( i,j ) = i+j;
      counter(0)++;
    }
  });

  //Initialize x vector on host using parallel_for
  Kokkos::parallel_for( range_policy (0, N), KOKKOS_LAMBDA ( const int i) {
    for (int j = 0; j < N; j++){
      x ( i,j ) = 2 * y ( i, j );
    }
  });

  for ( int i = 0; i < N; i++) {
    for ( int j = 0; j < N; j++) {
      ASSERT_EQ(x(i,j), 2 *(i+j));
    }
  }

  KokkosResilience::clear_duplicates_cache();

  ASSERT_EQ(counter(0), 3*N*N);

}

//Test MiniMD Exact Kernel Behavior
//Test worked, then was rewritten to demonstrate failure due to read/write copy error
//Copy-in-place and overwrite original with result leads to massive error
TEST(TestResOpenMP, TestMiniMDKernel)
{

  // Allocate 2D y, x vectors.
  Kokkos::View<double*[2],Kokkos::LayoutRight, Kokkos::HostSpace > x( "x", N );
  Kokkos::View<double*[2],Kokkos::LayoutRight, Kokkos::HostSpace > y( "y", N );
  Kokkos::View<double*[2],Kokkos::LayoutRight, Kokkos::HostSpace > z( "z", N );

  //Initialize x vector REGULAR kernel
  Kokkos::parallel_for( range_policy2 (0, N), KOKKOS_LAMBDA ( const int i) {
     x ( i,0 ) = 1;
  });

  int j = 0;

  while (j<5){
    //Test MiniMD Kernel Behavior with RESILIENT kernel, NONRESILEINT views
    Kokkos::parallel_for( range_policy (0, N), KOKKOS_LAMBDA ( const int i) {
      y ( i, 0 ) += test_const * x ( i, 0 );
      z ( i, 0 ) += test_const * y ( i, 0 );
    });
    j++;
  }

  KokkosResilience::clear_duplicates_cache();

  std::cout << "Test values y(1,0) and z(1,0) are " << y(1,0) << " and " << z(1,0) << " respectively\n. ";
  std::cout << "This test should have resulted in 50 and 1500, in these integrations, respectively." << std::endl;

  for ( int i = 0; i < N; i++) {
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
  KokkosResilience::global_error_settings = KokkosResilience::Error(0.005);

  // Allocate 2D y, x vectors.
  Kokkos::View<double*[2],Kokkos::LayoutRight, MemSpace,
               Kokkos::Experimental::SubscribableViewHooks<
               KokkosResilience::ResilientDuplicatesSubscriber >> x( "x", N );
  Kokkos::View<double*[2],Kokkos::LayoutRight, MemSpace,
               Kokkos::Experimental::SubscribableViewHooks<
               KokkosResilience::ResilientDuplicatesSubscriber >> y( "y", N );
  Kokkos::View<double*[2],Kokkos::LayoutRight, MemSpace,
               Kokkos::Experimental::SubscribableViewHooks<
               KokkosResilience::ResilientDuplicatesSubscriber >> z( "z", N );

  //Initialize x vector RESIIENT kernel WITH ERRORS
  Kokkos::parallel_for( range_policy (0, N), KOKKOS_LAMBDA ( const int i) {
     x ( i,0 ) = 1;
  });

  int j = 0;

  while (j<5){
    //Test MiniMD Kernel Behavior with RESILIENT kernel, RESILEINT views WITH ERRORS (cont prev count)
    Kokkos::parallel_for( range_policy (0, N), KOKKOS_LAMBDA ( const int i) {
      y ( i, 0 ) += test_const * x ( i, 0 );
      z ( i, 0 ) += test_const * y ( i, 0 );
    });
    j++;
  }

  for ( int i = 0; i < N; i++) {
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
  Kokkos::View<double*, Kokkos::LayoutRight, MemSpace,
               Kokkos::Experimental::SubscribableViewHooks<
                 KokkosResilience::ResilientDuplicatesSubscriber >,
		 Kokkos::MemoryTraits <Kokkos::RandomAccess> > x( "x", N );
  Kokkos::View<double*, Kokkos::LayoutRight, MemSpace,
               Kokkos::Experimental::SubscribableViewHooks<
                 KokkosResilience::ResilientDuplicatesSubscriber >, 
                 Kokkos::MemoryTraits <Kokkos::RandomAccess> > y( "y", N );
 
  Kokkos::parallel_for( range_policy (0, N), KOKKOS_LAMBDA ( const int i) {
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

  Kokkos::View<double*, Kokkos::LayoutRight, MemSpace,
               Kokkos::Experimental::SubscribableViewHooks<
                 KokkosResilience::ResilientDuplicatesSubscriber >,
                 Kokkos::MemoryTraits <Kokkos::Atomic> > x( "x", N );
  Kokkos::View<double*, Kokkos::LayoutRight, MemSpace,
               Kokkos::Experimental::SubscribableViewHooks<
                 KokkosResilience::ResilientDuplicatesSubscriber >,
                 Kokkos::MemoryTraits <Kokkos::Atomic> > y( "y", N );

  Kokkos::View<int*, Kokkos::LayoutRight, MemSpace,
               Kokkos::Experimental::SubscribableViewHooks<
                 KokkosResilience::ResilientDuplicatesSubscriber >,
                 Kokkos::MemoryTraits <Kokkos::Atomic> > counter( "counter", 1);
  Kokkos::View<int*, Kokkos::LayoutRight, Kokkos::HostSpace> counter2( "counter2", 1);

  Kokkos::parallel_for( range_policy (0, N), KOKKOS_LAMBDA ( const int i) {
    y ( i ) = i;
    counter(0)++;
    Kokkos::atomic_increment(&counter2(0));
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

/*
// Test Unmanaged
TEST(TestResOpenMP, TestUnmanaged)
{
  // Allocate Unmanaged Subscribers x, y vectors. 
  // NO LABELS on unmanged.
  // Should suppress reference counting.
  // Pointer to outside data needed?
  Kokkos::View<double*, Kokkos::LayoutRight, MemSpace,
               Kokkos::Experimental::SubscribableViewHooks<
                 KokkosResilience::ResilientDuplicatesSubscriber >,
                 Kokkos::MemoryTraits <Kokkos::Unmanaged> > x( N );
  Kokkos::View<double*, Kokkos::LayoutRight, MemSpace,
               Kokkos::Experimental::SubscribableViewHooks<
                 KokkosResilience::ResilientDuplicatesSubscriber >, 
                 Kokkos::MemoryTraits <Kokkos::Unmanaged> > y( N );

  Kokkos::Timer timer;

  Kokkos::parallel_for( range_policy (0, N), KOKKOS_LAMBDA ( const int i) {
    y ( i ) = i;
  });

  KokkosResilience::clear_duplicates_cache();

  std::cout << std::endl;

  Kokkos::deep_copy(x, y);
  for ( int i = 0; i < N; i++) {
    ASSERT_EQ(x(i), i);
  }
}
*/
/*
// Test DualView in Host/Host space with resilient subscriber
TEST(TestResOpenMP, TestDualView)
{
  using res_dual_view_type = Kokkos::DualView< double*,
                                     Kokkos::LayoutLeft>
  

  res_dual_view_type x("x", N);
  res_dual_view_type y("y", N);

  //Device view copy
  Kokkos::deep_copy(a.d_view, 1);
  //Constructed from a resilient view in the first place. This will be in default space.
  y.template modify<typename res_dual_view_type::execution_space>();
  y.template sync<typename res_dual_view_type::host_mirror_space>();

  //Host view copy
  Kokkos::deep_copy(a.h_view, 2);
  y.template modify<typename ViewType::host_mirror_space>();
  y.template sync<typename ViewType::execution_space>();  
  

  

  Kokkos::parallel_for( range_policy (0, N), KOKKOS_LAMBDA ( const int i) {
     
    //
    //y ( i ) = i;
  });

  KokkosResilience::clear_duplicates_cache();

  std::cout << std::endl;

  Kokkos::deep_copy(x, y);
  for ( int i = 0; i < N; i++) {
    ASSERT_EQ(x(i), i);
  }

}*/

