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


#include <Kokkos_Macros.hpp>
#if defined(KOKKOS_ENABLE_HPX)
#include <hpx/hpx_main.hpp>
#endif

#include <Kokkos_Core.hpp>
#include <resilience/Resilience.hpp>
#include <mpi.h>

using chkpt_view = Kokkos::Experimental::SubscribableViewHooks<KokkosResilience::DynamicViewHooksSubscriber>;

int
main( int argc, char **argv )
{
  MPI_Init( &argc, &argv );
  
  Kokkos::initialize( argc, argv );
  {
    auto ctx = KokkosResilience::make_context( MPI_COMM_WORLD, "config_file.json" );

    int  dim0 = 5, dim1 = 5;
    auto view = Kokkos::View< double **, chkpt_view>( "test_view", dim0, dim1 );

    KokkosResilience::checkpoint( *ctx, "test_checkpoint", 0, [view, dim0, dim1]() {
      Kokkos::parallel_for( dim0, KOKKOS_LAMBDA( int i ) {
        for ( int j = 0; j < dim1; ++j )
          view( i, j ) = 3.0;
      } );
    }, [](int){return true;} );
    
    for(int i = 0; i < dim0; i++){
      for(int j = 0; j < dim1; j++){
        if(view(i,j) != 3.0) {
          fprintf(stderr, "Error: view(%d,%d) = %f, not %f\n", i, j, view(i,j), 3.0);
          exit(1);
        }
      }
    }
    printf("Success!\n");

  }
  Kokkos::finalize();
  
  MPI_Finalize();
}
