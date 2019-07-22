#ifndef INC_TEST_COMMON_HPP
#define INC_TEST_COMMON_HPP

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

using enabled_exec_spaces = ::testing::Types<
#ifdef KOKKOS_ENABLE_SERIAL
  Kokkos::Serial
#endif
#ifdef KOKKOS_ENABLE_OPENMP
  Kokkos::OpenMP
#endif
>;

#endif // INC_TEST_COMMON_HPP
