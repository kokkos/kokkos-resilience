
#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <resilience/Context.hpp>
#include <resilience/veloc/VelocBackend.hpp>
#include <resilience/AutomaticCheckpoint.hpp>

int
main( int argc, char **argv )
{
  MPI_Init( &argc, &argv );
  
  Kokkos::initialize( argc, argv );
  {
    auto ctx = KokkosResilience::make_context( MPI_COMM_WORLD, "config.json" );
    
    int  dim0 = 5, dim1 = 5;
    auto view = Kokkos::View< double ** >( "test_view", dim0, dim1 );
    
    KokkosResilience::checkpoint( *ctx, "test_checkpoint", 0, [view, dim0, dim1]() {
      Kokkos::parallel_for( dim0, KOKKOS_LAMBDA( int i ) {
        for ( int j = 0; j < dim1; ++j )
          view( i, j ) = 3.0;
      } );
    } );
    
  }
  Kokkos::finalize();
  
  MPI_Finalize();
}
