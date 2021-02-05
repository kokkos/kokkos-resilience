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