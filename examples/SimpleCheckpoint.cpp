
#include <mpi.h>
#include <Kokkos_Core.hpp>
#define KOKKOS_ENABLE_AUTOMATIC_CHECKPOINT
#include <resilience/Resilience.hpp>

int
main( int argc, char **argv )
{
  MPI_Init( &argc, &argv );
  
  Kokkos::initialize( argc, argv );
  {
    auto ctx = KokkosResilience::Context< KokkosResilience::VeloCMemoryBackend >( MPI_COMM_WORLD, "veloc_test.cfg" );
    
    int  dim0 = 5, dim1 = 5;
    auto view = Kokkos::View< double ** >( "test_view", dim0, dim1 );
    
    KokkosResilience::checkpoint( ctx, "test_checkpoint", 0, [view, dim0, dim1]() {
      Kokkos::parallel_for( dim0, KOKKOS_LAMBDA( int i ) {
        for ( int j = 0; j < dim1; ++j )
          view( i, j ) = 3.0;
      } );
    } );
    
  }
  Kokkos::finalize();
  
  MPI_Finalize();
}
