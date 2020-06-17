#include "TestCommon.hpp"

#include <resilience/veloc/VelocBackend.hpp>
#include <resilience/AutomaticCheckpoint.hpp>
#include <resilience/MPIContext.hpp>
#include <resilience/filesystem/Filesystem.hpp>

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
    
    auto e = std::default_random_engine( 0 );
    auto ud = std::uniform_real_distribution< double >( -10.0, 10.0 );
    
    using view_type = Kokkos::View< double **, Layout, memory_space >;
    
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
    
    KokkosResilience::remove_all( "data/scratch" );
    KokkosResilience::remove_all( "data/persistent" );
    KokkosResilience::create_directory( "data/scratch" );
    KokkosResilience::create_directory( "data/persistent" );
    
    KokkosResilience::checkpoint( ctx, "test_checkpoint", 0, [=]() {
      Kokkos::parallel_for( Kokkos::RangePolicy<exec_space>( 0, dimx ), KOKKOS_LAMBDA( int i ) {
        for ( int j = 0; j < dimy; ++j )
          main_view( i, j ) -= 1.0;
      } );
    } );

    // Clobber main_view, should be reloaded at checkpoint
    Kokkos::parallel_for( Kokkos::RangePolicy<exec_space>( 0, dimx ), KOKKOS_LAMBDA( int i ) {
      for ( int j = 0; j < dimy; ++j )
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
  cfg["backends"]["veloc"]["config"].set( "data/veloc_test.cfg"s );
  auto ctx = KokkosResilience::MPIContext< KokkosResilience::VeloCMemoryBackend >( MPI_COMM_WORLD, cfg );
  
  using exec_space = typename TestFixture::exec_space;
  using memory_space = typename exec_space::memory_space;
  
  for ( std::size_t dimx = 1; dimx < 5; ++dimx )
  {
    for ( std::size_t dimy = 1; dimy < 5; ++dimy )
    {
      TestFixture::template test_layout< Kokkos::LayoutRight >( ctx, dimx, dimy );
      TestFixture::template test_layout< Kokkos::LayoutLeft >( ctx, dimx, dimy );
    }
  }
  
}
