#include <resilience/checkpoint.hpp>
#include <resilience/veloc/veloc_backend.hpp>
#include <mpi.h>
#include <Kokkos_Core.hpp>

int
main( int argc, char **argv )
{
  MPI_Init( &argc, &argv );
  
  Kokkos::initialize( argc, argv );
  {
    
    KokkosResilience::VeloCCheckpointBackend checkpoint(MPI_COMM_WORLD, "veloc_test.cfg" );
    
    int  dim0 = 5, dim1 = 5;
    auto view = Kokkos::View< double ** >( "test_view", dim0, dim1 );
    
    auto hview = Kokkos::create_mirror_view( view );
    
    Kokkos::deep_copy( view, hview );
    
    KokkosResilience::checkpoint( "test_checkpoint", 0, [view, dim0, dim1]() {
      Kokkos::parallel_for( dim0, KOKKOS_LAMBDA( int i ) {
        for ( int j = 0; j < dim1; ++j )
          view( i, j ) = 3.0;
      } );
    }, checkpoint );
    
  }
  Kokkos::finalize();
  
  MPI_Finalize();
}
