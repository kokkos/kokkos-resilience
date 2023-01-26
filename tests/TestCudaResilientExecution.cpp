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
using ResVectorType = Kokkos::View<double*, Kokkos::LayoutLeft, MemSpace>;

//Non-resilient
using ViewVectorType = Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::CudaSpace>;
using range_policy2 = Kokkos::RangePolicy<Kokkos::Cuda>;

// gTest runs CUDA resilient execution space and performs a deep copy to show space
// itself is working.
TEST(TestResCuda, TestCudaSpace)
{
  
  // Create device view
  ResVectorType x( "x", N );
  ResVectorType y( "y", N );

  // Create host mirror of device view
  ResVectorType::HostMirror h_x = Kokkos::create_mirror_view( x );
  auto h_y = Kokkos::create_mirror_view( x );

  // Initialize x vector on host
  for ( int i = 0; i < N; i++ )
    h_y( i ) = i;

  // Deep copy device to host (dest, src), checking if space works
  // Does not use parallel_for code, should pass deep-copy from ResCudaSpace to CudaSpace
  Kokkos::deep_copy( x, h_x );
 
}

/*********************************
*********PARALLEL FORS************
**********************************/

void test_kokkos_for(){

  // Create device views
  ViewVectorType y( "y", N );
  ViewVectorType x( "x", N );

  // Create host mirrors of device views.
  ViewVectorType::HostMirror h_y = Kokkos::create_mirror_view( y );
  ViewVectorType::HostMirror h_x = Kokkos::create_mirror_view( x );

  // Initialize y vector on host
  for ( int i = 0; i < N; i++ ){
    h_y( i ) = i;
  }

  // Deep copy host to device (dest, src)
  Kokkos::deep_copy( y, h_y );

  // Copy y to x on device using parallel for
  Kokkos::parallel_for(
      range_policy2(0, N), KOKKOS_LAMBDA(int i) { x (i) = y (i) * y (i) ; });

  // Deep copy device to host (dest, src)
  Kokkos::deep_copy(h_x, x);
  
  // Assert that the parallel for took
  for ( int i = 0; i < N; i++) {
    ASSERT_EQ(h_x(i), i*i);
  }

}

// gTest runs CUDA parallel_for with non-resilient Kokkos. Should never fail.
TEST(TestResOpenMP, TestKokkosFor)
{
 
  test_kokkos_for();

}


