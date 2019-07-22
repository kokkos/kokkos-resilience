
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

int
main( int argc, char **argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  
  Kokkos::initialize( argc, argv );
  
  auto ret = RUN_ALL_TESTS();
  
  Kokkos::finalize();
  
  return ret;
}