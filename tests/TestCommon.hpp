#ifndef INC_TEST_COMMON_HPP
#define INC_TEST_COMMON_HPP

#include <Kokkos_Macros.hpp>
#if defined(KOKKOS_ENABLE_HPX)
#include <hpx/config.hpp>
#endif
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

namespace detail
{
  struct dummy {};
  
  template< typename, typename... Rest >
  using remove_first_type = ::testing::Types< Rest... >;
  
  using exec_spaces = remove_first_type< dummy
#ifdef KOKKOS_ENABLE_SERIAL
    , Kokkos::Serial
#endif
#ifdef KOKKOS_ENABLE_OPENMP
    , Kokkos::OpenMP
#endif
#ifdef KOKKOS_ENABLE_CUDA
    , Kokkos::Cuda
#endif
#ifdef KOKKOS_ENABLE_HPX
    , Kokkos::Experimental::HPX
#endif  
  >;
}

using enabled_exec_spaces = detail::exec_spaces;

#endif // INC_TEST_COMMON_HPP
