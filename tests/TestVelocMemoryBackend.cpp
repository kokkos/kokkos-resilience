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
#include "TestCommon.hpp"

#include <resilience/backend/VelocBackend.hpp>
#include <resilience/AutomaticCheckpoint.hpp>
#include <resilience/context/MPIContext.hpp>
#include <resilience/util/filesystem/Filesystem.hpp>

#include <string>

#include <random>

template< typename ExecSpace >
class TestVelocMemoryBackend : public ::testing::Test
{
public:

  using exec_space = ExecSpace;

  template< typename Layout, typename Context >
  static void test_layout( Context &ctx, std::size_t dimx, std::size_t dimy )
  {
    ctx.backend().reset();
    using memory_space = typename exec_space::memory_space;
    using view_type = KokkosResilience::View< double **, Layout, memory_space >;

    auto e = std::default_random_engine( 0 );
    auto ud = std::uniform_real_distribution< double >( -10.0, 10.0 );

    view_type main_view( "main_view", dimx, dimy );
    auto host_mirror = Kokkos::create_mirror_view( main_view );

    for ( std::size_t x = 0; x < dimx; ++x )
    {
      for ( std::size_t y = 0; y < dimy; ++y )
      {
        host_mirror( x, y ) = ud( e );
      }
    }

    Kokkos::deep_copy( main_view, host_mirror );

    Kokkos::fence();

    KokkosResilience::remove_all( KR_TEST_DATADIR "/scratch" );
    KokkosResilience::remove_all( KR_TEST_DATADIR "/persistent" );
    KokkosResilience::create_directory( KR_TEST_DATADIR "/scratch" );
    KokkosResilience::create_directory( KR_TEST_DATADIR "/persistent" );

    KokkosResilience::checkpoint( ctx, "test_checkpoint", 0, [=]() {
      Kokkos::parallel_for( Kokkos::RangePolicy<exec_space>( 0, dimx ), KOKKOS_LAMBDA( int i ) {
        for ( std::size_t j = 0; j < dimy; ++j )
          main_view( i, j ) -= 1.0;
      } );
    } );

    // Clobber main_view, should be reloaded at checkpoint
    Kokkos::parallel_for( Kokkos::RangePolicy<exec_space>( 0, dimx ), KOKKOS_LAMBDA( int i ) {
      for ( std::size_t j = 0; j < dimy; ++j )
        main_view( i, j ) = 0.0;
    } );

    // Clobber host view just in case

    for ( std::size_t x = 0; x < dimx; ++x )
    {
      for ( std::size_t y = 0; y < dimy; ++y )
      {
        host_mirror( x, y ) = 0.0;
      }
    }

    // The lambda shouldn't be executed, instead recovery should start
    KokkosResilience::checkpoint( ctx, "test_checkpoint", 0, [main_view]() {
      ASSERT_TRUE( false );
    } );

    Kokkos::fence();

    Kokkos::deep_copy( host_mirror, main_view );

    e.seed( 0 );

    for ( std::size_t x = 0; x < dimx; ++x )
    {
      for ( std::size_t y = 0; y < dimy; ++y )
      {
        ASSERT_EQ( host_mirror( x, y ), ud( e ) - 1.0 );
      }
    }
  }
};


TYPED_TEST_SUITE( TestVelocMemoryBackend, enabled_exec_spaces );

TYPED_TEST( TestVelocMemoryBackend, veloc_mem )
{
  using namespace std::string_literals;
  KokkosResilience::Config cfg;
  cfg["backend"].set( "veloc"s );
  cfg["backends"]["veloc"]["config"].set( KR_TEST_DATADIR "/veloc_test.cfg" );
  KokkosResilience::MPIContext< KokkosResilience::VeloCMemoryBackend > ctx( MPI_COMM_WORLD, cfg );

  for ( std::size_t dimx = 1; dimx < 5; ++dimx )
  {
    for ( std::size_t dimy = 1; dimy < 5; ++dimy )
    {
      TestFixture::template test_layout< Kokkos::LayoutRight >( ctx, dimx, dimy );
      TestFixture::template test_layout< Kokkos::LayoutLeft >( ctx, dimx, dimy );
    }
  }

}
