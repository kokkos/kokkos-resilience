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
#include <resilience/Resilience.hpp>
#include <Kokkos_Core.hpp>

void iterate( KokkosResilience::ContextBase &ctx, int max_ts )
{
  int ns = KokkosResilience::latest_version( ctx, "iterate" );
  if ( ns < 0 )
    ns = 0;

  Kokkos::View< double * > ping( "ping", 1000 );
  Kokkos::View< double * > pong( "pong", 1000 );

  Kokkos::parallel_for( 1000, KOKKOS_LAMBDA( int i ) {
    ping( i ) = 0.0;
  } );

  Kokkos::parallel_for( 1000, KOKKOS_LAMBDA( int i ) {
    pong( i ) = -1.0;
  } );

  Kokkos::fence();

  for ( int i = ns; i < max_ts; ++i )
  {
    Kokkos::View< const double * > read;
    Kokkos::View< double * > write;
    if ( i % 2 )
    {
      read = pong;
      write = ping;
    } else {
      read = ping;
      write = pong;
    }
    std::cout << "reading " << read( 0 ) << " from " << read.label() << '\n';
    KokkosResilience::checkpoint( ctx, "iterate", i, [=]() {
      Kokkos::parallel_for( 1000, KOKKOS_LAMBDA( int i ) {
        write( i ) = read( i ) + 1.0;
      } );
    } );

    Kokkos::fence();

    std::cout << "wrote " << write( 0 ) << " to " << write.label() << '\n';
  }
}

int
main( int argc, char **argv )
{
  MPI_Init( &argc, &argv );

  Kokkos::initialize( argc, argv );
  {
    auto ctx = KokkosResilience::make_context( MPI_COMM_WORLD, "config.json" );
    iterate( *ctx, 1000 );
  }
  Kokkos::finalize();

  MPI_Finalize();
}