

#include <Kokkos_Macros.hpp>
#if defined(KOKKOS_ENABLE_HPX)
#include <hpx/hpx_main.hpp>
#endif

#include <Kokkos_Core.hpp>
#include <resilience/Context.hpp>
#include <resilience/stdfile/StdFileBackend.hpp>
#include <resilience/AutomaticCheckpoint.hpp>

int
main( int argc, char **argv )
{
  Kokkos::initialize( argc, argv );
  {
    auto ctx = KokkosResilience::make_context( "checkpoint.data", "config_file.json" );

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
}
