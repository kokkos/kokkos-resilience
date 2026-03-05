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
using KokkosCudaView = Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::CudaSpace>;
using cuda_range_policy = Kokkos::RangePolicy<Kokkos::Cuda>;

TEST(TestResCuda, TestCudaSpace)
{

  // Create device view
  KokkosCudaView x( "x", N );
  KokkosCudaView y( "y", N );

  std::cout << "Kokkos Test for SegFault after allocate device views" << std::endl;

  // Create host mirror of device view
  auto h_x = Kokkos::create_mirror( x );
  auto h_y = Kokkos::create_mirror( y );

  static_assert( std::same_as<decltype(h_x)::memory_space, Kokkos::HostSpace > );
  std::cout << "Kokkos Test for SegFault after allocate host mirror before init host PFor" << std::endl;

  // Initialize x vector on host
  for ( int i = 0; i < N; i++ ){
    h_x( i ) = i;
  }

  std::cout << "Kokkos Test for SegFault after Init host PFor" << std::endl;

  // Deep copy device to host (dest, src), checking if space works
  // Does not use parallel_for code, should pass deep-copy from ResCudaSpace to CudaSpace
  Kokkos::deep_copy( x, h_x );

  std::cout << "Kokkos Test for SegFault after deep-copy host-host" << std::endl;
  const std::type_info& ti = typeid(decltype(h_x)::memory_space);
  std::cout << "Get name? "<< ti.name() << std::endl;

  Kokkos::deep_copy ( y, x);
  Kokkos::deep_copy ( h_y, y);

  for (int i=0; i< N; i++) {
    ASSERT_EQ( h_y(i), i);
  }

  Kokkos::fence();

}
//#if 0
// gTest runs CUDA resilient execution space and performs a deep copy to show space
// itself is working.
TEST(TestResCuda, TestResCudaSpace)
{
  
  // Create device view
  ResilientView<double*> x( Kokkos::view_alloc("x", Kokkos::WithoutInitializing), N );
  ResilientView<double*> y( Kokkos::view_alloc("y", Kokkos::WithoutInitializing), N );
  
  //ResilientView<double*> x( "x", N );
  //ResilientView<double*> y( "y", N );

  std::cout << "Test for SegFault after allocate device views" << std::endl;

  // Create host mirror of device view
  auto h_x = Kokkos::create_mirror( x );
  auto h_y = Kokkos::create_mirror( y );

  const std::type_info& ti = typeid(decltype(h_x)::memory_space);
  std::cout << "Get name? "<< ti.name() << std::endl;
  //static_assert( std::same_as<decltype(h_x)::memory_space, Kokkos::HostSpace > );
  std::cout << "Test for SegFault after allocate host mirror before init host PFor" << std::endl;

  // Initialize x vector on host
  for ( int i = 0; i < N; i++ ){
    h_x( i ) = i;
  }

  std::cout << "Test for SegFault after Init host PFor" << std::endl;

  // Deep copy device to host (dest, src), checking if space works
  // Does not use parallel_for code, should pass deep-copy from ResCudaSpace to CudaSpace
  Kokkos::deep_copy( x, h_x );

  std::cout << "Test for SegFault after deep-copy host-host" << std::endl;

  Kokkos::fence();

  KokkosResilience::clear_duplicates_cache();

}
//#endif
/*********************************
*********PARALLEL FORS************
**********************************/

void test_kokkos_for(){

  // Create device views
  KokkosCudaView y( "y", N );
  KokkosCudaView x( "x", N );

  // Create host mirrors of device views.
  auto h_y = Kokkos::create_mirror_view( y );
  auto h_x = Kokkos::create_mirror_view( x );
//KokkosCudaView::HostMirror h_x = Kokkos::create_mirror_view( x );

  // Initialize y vector on host
  for ( int i = 0; i < N; i++ ){
    h_y( i ) = i;
  }

  // Deep copy host to device (dest, src)
  Kokkos::deep_copy( y, h_y );

  // Copy y to x on device using parallel for
  Kokkos::parallel_for(cuda_range_policy(0, N), KOKKOS_LAMBDA(int i) { 
	  x (i) = y (i) * y (i); 
	  });

  // Deep copy device to host (dest, src)
  Kokkos::deep_copy(h_x, x);
  
  // Assert that the parallel for took
  for ( int i = 0; i < N; i++) {
    ASSERT_EQ(h_x(i), i*i);
  }

}

// gTest runs CUDA parallel_for with non-resilient Kokkos. Should never fail.
TEST(TestResCuda, TestKokkosFor)
{
 
  test_kokkos_for();

}

template class Kokkos::Impl::ParallelFor<KokkosResilience::TestKernel<ResilientView<double*>>, cuda_range_policy, Kokkos::Cuda>;

void test_res_for(){

  //ResilientView<double*> x( Kokkos::view_alloc("x", Kokkos::WithoutInitializing), N );
  //ResilientView<double*> y( Kokkos::view_alloc("y", Kokkos::WithoutInitializing), N );

  ResilientView<double*> x( "x", N );
  ResilientView<double*> y( "y", N );

  std::cout << "Reached after Initialized cuda views." << std::endl;

  // Create host mirror of device view
  auto h_x = Kokkos::create_mirror( x );
  auto h_y = Kokkos::create_mirror( y );

  const std::type_info& ti = typeid(decltype(h_x)::memory_space);
  std::cout << "Memory space of host mirror (created): " << ti.name() << std::endl;

  // Initialize y vector on host
  for ( int i = 0; i < N; i++ ){
    h_y( i ) = i;
  }

  // Deep copy host to device (dest, src)
  Kokkos::deep_copy( y, h_y );

  std::cout << "********After deep-copy host->device, before for" << std::endl;
  // Basic resilient parallel for

 // Kokkos::parallel_for(cuda_range_policy(0,N), KokkosResilience::TestKernel{.view=y});
  //Kokkos::fence();
  
#if 1

  Kokkos::parallel_for(res_range_policy(0, N), KOKKOS_LAMBDA(int i) {
          x (i) = y (i) * y (i);
          });
  std::cout << "*********After resilient parallel-for" << std::endl;
#endif
  // Deep copy device to host (dest, src)
  Kokkos::deep_copy(h_x, x);

  KokkosResilience::clear_duplicates_cache();

  // Assert that the parallel for took
  for ( int i = 0; i < N; i++) {
    ASSERT_EQ(h_x(i), i*i);
  }
}

// gTest runs CUDA parallel_for with non-resilient Kokkos. Should never fail.
TEST(TestResCuda, TestResFor)
{

  test_res_for();	

}
