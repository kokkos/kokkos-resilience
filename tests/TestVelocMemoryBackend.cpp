// This is only needed for gtest...
// #include "TestCommon.hpp"  

#if defined(KR_ENABLE_VELOC)
#include <resilience/veloc/VelocBackend.hpp>
#include <resilience/AutomaticCheckpoint.hpp>
#include <resilience/Context.hpp>

/* this is only needed for gtest
template< typename ExecSpace >
class TestVelocMemoryBackend
{
public:
  
  using exec_space = ExecSpace;
};
*/

int
main( int argc, char **argv )
{
MPI_Init(&argc, &argv);
Kokkos::initialize(argc,argv);
  auto ctx = KokkosResilience::Context< KokkosResilience::VeloCMemoryBackend >( MPI_COMM_WORLD, "veloc_test.cfg" );
  
  int  dim0 = 5, dim1 = 5;
  auto view = Kokkos::View< double ** >( "test_view", dim0, dim1 );
  
  KokkosResilience::checkpoint( ctx, "test_checkpoint", 0, [view, dim0, dim1]() {
    Kokkos::parallel_for( dim0, KOKKOS_LAMBDA( int i ) {
      for ( int j = 0; j < dim1; ++j )
        view( i, j ) = 3.0;
    } );
  } );
Kokkos::finalize();
MPI_Finalize();
}
#endif
