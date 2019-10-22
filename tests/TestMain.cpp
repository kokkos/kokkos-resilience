
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <resilience/Resilience.hpp>

int
main( int argc, char **argv )
{
  ::testing::InitGoogleTest( &argc, argv );
#if defined(KR_ENABLE_HDF5_PARALLEL) || defined(KR_ENABLE_VELOC)
  MPI_Init( &argc, &argv );
#endif
  
  Kokkos::initialize( argc, argv );
  
  auto ret = RUN_ALL_TESTS();
  
  Kokkos::finalize();

#if defined(KR_ENABLE_HDF5_PARALLEL) || defined(KR_ENABLE_VELOC)
  MPI_Finalize();
#endif
  
  return ret;
}