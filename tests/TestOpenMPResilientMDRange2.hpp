#include <cstdio>
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <resilience/openMP/ResHostSpace.hpp>
#include <resilience/openMP/ResOpenMP.hpp>

#define MemSpace KokkosResilience::ResHostSpace
#define ExecSpace2 KokkosResilience::ResOpenMP

namespace Test {

namespace {

using namespace Kokkos;

template <typename ExecSpace>
struct TestMDRange_2D {
  using DataType     = int;
  using ViewType     = typename Kokkos::View<DataType **, ExecSpace,
                                             Kokkos::Experimental::SubscribableViewHooks<
                                             KokkosResilience::ResilientDuplicatesSubscriber > >;
  using HostViewType = typename ViewType::HostMirror;

  ViewType input_view;
  using value_type = double;

  TestMDRange_2D(const DataType N0, const DataType N1)
      : input_view("input_view", N0, N1) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, const int j) const { input_view(i, j) = 1; }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, const int j, value_type &lsum) const {
    lsum += input_view(i, j) * 2;
  }
  // tagged operators
  struct InitTag {};
  KOKKOS_INLINE_FUNCTION
  void operator()(const InitTag &, const int i, const int j) const {
    input_view(i, j) = 3;
  }

  // reduction tagged operators
  KOKKOS_INLINE_FUNCTION
  void operator()(const InitTag &, const int i, const int j,
                  value_type &lsum) const {
    lsum += input_view(i, j) * 3;
  }

  static void test_for2(const int N0, const int N1) {
#if defined(KOKKOS_ENABLE_CXX11_DISPATCH_LAMBDA)
    {
      using range_type =
          typename Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>,
                                         Kokkos::IndexType<int>>;
      using tile_type  = typename range_type::tile_type;
      using point_type = typename range_type::point_type;

      const int s0 = 1;
      const int s1 = 1;

      range_type range(point_type{{s0, s1}}, point_type{{N0, N1}},
                       tile_type{{3, 3}});

      TestMDRange_2D::ViewType v("v", N0, N1);

      parallel_for(
          range, KOKKOS_LAMBDA(const int i, const int j) { v(i, j) = 3; });

      TestMDRange_2D::HostViewType h_view = Kokkos::create_mirror_view(v);
      Kokkos::deep_copy(h_view, v);

      int counter = 0;
      for (int i = s0; i < N0; ++i)
        for (int j = s1; j < N1; ++j) {
          if (h_view(i, j) != 3) {
            ++counter;
          }
        }

      if (counter != 0) {
        printf(
            "Offset Start + Default Layouts + InitTag op(): Errors in "
            "test_for2; mismatches = %d\n\n",
            counter);
      }

      ASSERT_EQ(counter, 0);
    }
#endif

    {
      using range_type =
          typename Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>,
                                         Kokkos::IndexType<int>, InitTag>;
      using tile_type  = typename range_type::tile_type;
      using point_type = typename range_type::point_type;

      const int s0 = 1;
      const int s1 = 1;
      range_type range(point_type{{s0, s1}}, point_type{{N0, N1}},
                       tile_type{{3, 3}});
      TestMDRange_2D functor(N0, N1);

      parallel_for(range, functor);

      HostViewType h_view = Kokkos::create_mirror_view(functor.input_view);
      Kokkos::deep_copy(h_view, functor.input_view);

      int counter = 0;
      for (int i = s0; i < N0; ++i)
        for (int j = s1; j < N1; ++j) {
          if (h_view(i, j) != 3) {
            ++counter;
          }
        }

      if (counter != 0) {
        printf(
            "Offset Start + Default Layouts + InitTag op(): Errors in "
            "test_for2; mismatches = %d\n\n",
            counter);
      }

      ASSERT_EQ(counter, 0);
    }

    {
      using range_type =
          typename Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>, InitTag>;
      using tile_type  = typename range_type::tile_type;
      using point_type = typename range_type::point_type;

      range_type range(point_type{{0, 0}}, point_type{{N0, N1}},
                       tile_type{{3, 3}});
      TestMDRange_2D functor(N0, N1);

      parallel_for(range, functor);

      HostViewType h_view = Kokkos::create_mirror_view(functor.input_view);
      Kokkos::deep_copy(h_view, functor.input_view);

      int counter = 0;
      for (int i = 0; i < N0; ++i)
        for (int j = 0; j < N1; ++j) {
          if (h_view(i, j) != 3) {
            ++counter;
          }
        }

      if (counter != 0) {
        printf(
            "Default Layouts + InitTag op(): Errors in test_for2; mismatches = "
            "%d\n\n",
            counter);
      }

      ASSERT_EQ(counter, 0);
    }

    {
      using range_type =
          typename Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>, InitTag>;
      using point_type = typename range_type::point_type;

      range_type range(point_type{{0, 0}}, point_type{{N0, N1}});
      TestMDRange_2D functor(N0, N1);

      parallel_for(range, functor);

      HostViewType h_view = Kokkos::create_mirror_view(functor.input_view);
      Kokkos::deep_copy(h_view, functor.input_view);

      int counter = 0;
      for (int i = 0; i < N0; ++i)
        for (int j = 0; j < N1; ++j) {
          if (h_view(i, j) != 3) {
            ++counter;
          }
        }

      if (counter != 0) {
        printf(
            "Default Layouts + InitTag op() + Default Tile: Errors in "
            "test_for2; mismatches = %d\n\n",
            counter);
      }

      ASSERT_EQ(counter, 0);
    }

    {
      using range_type =
          typename Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>,
                                         Kokkos::IndexType<int>>;
      using tile_type  = typename range_type::tile_type;
      using point_type = typename range_type::point_type;

      range_type range(point_type{{0, 0}}, point_type{{N0, N1}},
                       tile_type{{3, 3}});
      TestMDRange_2D functor(N0, N1);

      parallel_for(range, functor);

      HostViewType h_view = Kokkos::create_mirror_view(functor.input_view);
      Kokkos::deep_copy(h_view, functor.input_view);

      int counter = 0;
      for (int i = 0; i < N0; ++i)
        for (int j = 0; j < N1; ++j) {
          if (h_view(i, j) != 1) {
            ++counter;
          }
        }

      if (counter != 0) {
        printf("No info: Errors in test_for2; mismatches = %d\n\n", counter);
      }

      ASSERT_EQ(counter, 0);
    }

    {
      using range_type = typename Kokkos::MDRangePolicy<
          ExecSpace, Kokkos::Rank<2, Iterate::Default, Iterate::Default>,
          Kokkos::IndexType<int>>;
      using tile_type  = typename range_type::tile_type;
      using point_type = typename range_type::point_type;

      range_type range(point_type{{0, 0}}, point_type{{N0, N1}},
                       tile_type{{4, 4}});
      TestMDRange_2D functor(N0, N1);

      parallel_for(range, functor);

      HostViewType h_view = Kokkos::create_mirror_view(functor.input_view);
      Kokkos::deep_copy(h_view, functor.input_view);

      int counter = 0;
      for (int i = 0; i < N0; ++i)
        for (int j = 0; j < N1; ++j) {
          if (h_view(i, j) != 1) {
            ++counter;
          }
        }

      if (counter != 0) {
        printf("D D: Errors in test_for2; mismatches = %d\n\n", counter);
      }

      ASSERT_EQ(counter, 0);
    }

    {
      using range_type = typename Kokkos::MDRangePolicy<
          ExecSpace, Kokkos::Rank<2, Iterate::Left, Iterate::Left>,
          Kokkos::IndexType<int>>;
      using tile_type  = typename range_type::tile_type;
      using point_type = typename range_type::point_type;

      range_type range(point_type{{0, 0}}, point_type{{N0, N1}},
                       tile_type{{3, 3}});
      TestMDRange_2D functor(N0, N1);

      parallel_for(range, functor);

      HostViewType h_view = Kokkos::create_mirror_view(functor.input_view);
      Kokkos::deep_copy(h_view, functor.input_view);

      int counter = 0;
      for (int i = 0; i < N0; ++i)
        for (int j = 0; j < N1; ++j) {
          if (h_view(i, j) != 1) {
            ++counter;
          }
        }

      if (counter != 0) {
        printf("L L: Errors in test_for2; mismatches = %d\n\n", counter);
      }

      ASSERT_EQ(counter, 0);
    }

    {
      using range_type = typename Kokkos::MDRangePolicy<
          ExecSpace, Kokkos::Rank<2, Iterate::Left, Iterate::Right>,
          Kokkos::IndexType<int>>;
      using tile_type  = typename range_type::tile_type;
      using point_type = typename range_type::point_type;

      range_type range(point_type{{0, 0}}, point_type{{N0, N1}},
                       tile_type{{7, 7}});
      TestMDRange_2D functor(N0, N1);

      parallel_for(range, functor);

      HostViewType h_view = Kokkos::create_mirror_view(functor.input_view);
      Kokkos::deep_copy(h_view, functor.input_view);

      int counter = 0;
      for (int i = 0; i < N0; ++i)
        for (int j = 0; j < N1; ++j) {
          if (h_view(i, j) != 1) {
            ++counter;
          }
        }

      if (counter != 0) {
        printf("L R: Errors in test_for2; mismatches = %d\n\n", counter);
      }

      ASSERT_EQ(counter, 0);
    }

    {
      using range_type = typename Kokkos::MDRangePolicy<
          ExecSpace, Kokkos::Rank<2, Iterate::Right, Iterate::Left>,
          Kokkos::IndexType<int>>;
      using tile_type  = typename range_type::tile_type;
      using point_type = typename range_type::point_type;

      range_type range(point_type{{0, 0}}, point_type{{N0, N1}},
                       tile_type{{16, 16}});
      TestMDRange_2D functor(N0, N1);

      parallel_for(range, functor);

      HostViewType h_view = Kokkos::create_mirror_view(functor.input_view);
      Kokkos::deep_copy(h_view, functor.input_view);

      int counter = 0;
      for (int i = 0; i < N0; ++i)
        for (int j = 0; j < N1; ++j) {
          if (h_view(i, j) != 1) {
            ++counter;
          }
        }

      if (counter != 0) {
        printf("R L: Errors in test_for2; mismatches = %d\n\n", counter);
      }

      ASSERT_EQ(counter, 0);
    }

    {
      using range_type = typename Kokkos::MDRangePolicy<
          ExecSpace, Kokkos::Rank<2, Iterate::Right, Iterate::Right>,
          Kokkos::IndexType<int>>;
      using tile_type  = typename range_type::tile_type;
      using point_type = typename range_type::point_type;

      range_type range(point_type{{0, 0}}, point_type{{N0, N1}},
                       tile_type{{5, 16}});
      TestMDRange_2D functor(N0, N1);

      parallel_for(range, functor);

      HostViewType h_view = Kokkos::create_mirror_view(functor.input_view);
      Kokkos::deep_copy(h_view, functor.input_view);

      int counter = 0;
      for (int i = 0; i < N0; ++i)
        for (int j = 0; j < N1; ++j) {
          if (h_view(i, j) != 1) {
            ++counter;
          }
        }

      if (counter != 0) {
        printf("R R: Errors in test_for2; mismatches = %d\n\n", counter);
      }

      ASSERT_EQ(counter, 0);
    }

  }  // end test_for2
};   // MDRange_2D

}

}
