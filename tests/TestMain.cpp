
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <resilience/Resilience.hpp>

int
main( int argc, char **argv )
{
  ::testing::InitGoogleTest( &argc, argv );
#ifdef KOKKOS_ENABLE_HDF5_PARALLEL
  MPI_Init( &argc, &argv );
#endif
  
  Kokkos::initialize( argc, argv );
  
  auto ret = RUN_ALL_TESTS();
  
  Kokkos::finalize();

#ifdef KOKKOS_ENABLE_HDF5_PARALLEL
  MPI_Finalize();
#endif
  
  return ret;
}