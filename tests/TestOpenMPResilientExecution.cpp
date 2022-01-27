#include <gtest/gtest.h>
#include "TestCommon.hpp"
#include <Kokkos_Core.hpp>
#include <Kokkos_Core_fwd.hpp>
#include <resilience/Resilience.hpp>
#include <resilience/openMP/ResHostSpace.hpp>
#include <resilience/openMP/ResOpenMP.hpp>
//#include <resilience/openMP/OpenMPResSubscriber.hpp>

#include <ctime>
#include <random>
#include <time.h>
#include <thread>

#include <vector>
#include <math.h>

#include <omp.h>

// Take out when separated
#include <cstdio>
//#include <TestOpenMPResilientMDRange.hpp>

//#ifdef KR_ENABLE_OPENMP

#define N 25
#define MemSpace KokkosResilience::ResHostSpace
#define ExecSpace2 KokkosResilience::ResOpenMP
/*
namespace Test {

namespace {

using namespace Kokkos;

template <typename ExecSpace>
struct TestMDRange_2D {
  using DataType     = int;
  using ViewType     = typename Kokkos::View<DataType **, ExecSpace>;
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

// gTest resilient spaces work on own. Goal is to deepcopy one view to another.
TEST(TestResOpenMP, TestSpaces)
{
  Kokkos::fence();

  using ViewVectorType = Kokkos::View<double*, Kokkos::LayoutRight, MemSpace>;

  ViewVectorType y( "y", N);
  ViewVectorType x( "x", N);

  for ( int i = 0; i < N; i++ ) {
    y( i ) = 1;
  }

  Kokkos::Timer timer;
  Kokkos::deep_copy(x, y);
  double time = timer.seconds();

  std::cout << "The deep-copy took " << time << " seconds." << std::endl;

  for ( int i = 0; i < N; i++) {
    ASSERT_EQ(x(i), 1);
  }

  std::cout << std::endl << std::endl;

}
*/
/*********************************
*********PARALLEL FORS************
**********************************/
 ///*
// gTest runs parallel_for with non-resilient Kokkos. Should never fail.
TEST(TestResOpenMP, TestKokkosFor)
{

  std::cout << "KokkosFor Test" << std::endl;

  using range_policy2 = Kokkos::RangePolicy<Kokkos::OpenMP>;
  using ViewVectorType2 = Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::HostSpace>;

  // Allocate y, x vectors.
  ViewVectorType2 y2( "y", N );
  ViewVectorType2 x2( "x", N );

  Kokkos::Timer timer;

  Kokkos::fence();

  // Initialize y vector on host using parallel_for
  Kokkos::parallel_for( range_policy2 (0, N), KOKKOS_LAMBDA ( int i) {
    y2 ( i ) = i;
  });

  Kokkos::fence();
  double time = timer.seconds();

  Kokkos::deep_copy(x2, y2);
  for ( int i = 0; i < N; i++) {
    ASSERT_EQ(x2(i), i);
  }

  printf("GTEST: Thread %d reports Kokkos parallel_for took %f seconds.\n", omp_get_thread_num(), time);

  printf("\n\n\n");
  fflush(stdout);

}

// gTest runs parallel_for with resilient Kokkos doubles assignment and atomic counter.
// Expect counter to count iterations.
TEST(TestResOpenMP, TestResilientForDouble)
{

  std::cout << "KokkosResilient For Doubles" << std::endl;

  // range policy with resilient execution space
  using range_policy = Kokkos::RangePolicy<ExecSpace2>;

  using subscriber_vector_int_type = Kokkos::View< int* , Kokkos::LayoutRight, MemSpace,
      Kokkos::Experimental::SubscribableViewHooks<
          KokkosResilience::ResilientDuplicatesSubscriber > >;
  using subscriber_vector_double_type = Kokkos::View< double* , Kokkos::LayoutRight, MemSpace,
      Kokkos::Experimental::SubscribableViewHooks<
          KokkosResilience::ResilientDuplicatesSubscriber > >;

  // Allocate y, x vectors.
  subscriber_vector_double_type y( "y", N );
  subscriber_vector_double_type x( "x", N );

  printf("GTEST: Thread %d reports doubles vectors declared.\n", omp_get_thread_num());
  fflush(stdout);

  //Integer vector 1 long to count data accesses, because scalar view bugs (previously)
  subscriber_vector_int_type counter( "DataAccesses", 1);

  Kokkos::Timer timer;

  counter(0) = 0;

  printf("GTEST: Thread %d reports counter successfully initialized to %d.\n", omp_get_thread_num(), counter(0));
  fflush(stdout);

  //Initialize y vector on host using parallel_for, increment a counter for data accesses.
  Kokkos::parallel_for( range_policy (0, N), KOKKOS_LAMBDA ( const int i) {
    y ( i ) = i;
    //printf("This is i from the parallel for %d \n", i);
    //fflush(stdout);
    Kokkos::atomic_increment(&counter(0));
    //printf("This is counter(0): %d\n", counter(0));
    //fflush(stdout);

  });

  Kokkos::fence();

  double time = timer.seconds();
  printf("GTEST: Thread %d reports counter is %d. It should be %d.\n", omp_get_thread_num(), counter(0), N);
  fflush(stdout);

  Kokkos::deep_copy(x, y);

  for ( int i = 0; i < N; i++) {
    ASSERT_EQ(x(i), i);
  }

  printf("GTEST: Thread %d reports test parallel_for completed. Data assignment was correct.\n", omp_get_thread_num());
  fflush(stdout);

  ASSERT_EQ(counter(0), N);

  printf("\n\n\n");
  fflush(stdout);
}

// gTest runs parallel_for with resilient Kokkos integer assignment and atomic counter.
// Expect counter to count iterations.
TEST(TestResOpenMP, TestResilientForInteger)
{

    std::cout << "KokkosResilient For Integers" << std::endl;
    // range policy with resilient execution space
    using range_policy = Kokkos::RangePolicy<ExecSpace2>;

    using subscriber_vector_int_type = Kokkos::View< int* , Kokkos::LayoutRight, MemSpace,
            Kokkos::Experimental::SubscribableViewHooks<
                    KokkosResilience::ResilientDuplicatesSubscriber > >;


    // Allocate y, x vectors.
    subscriber_vector_int_type y( "y", N );
    subscriber_vector_int_type x( "x", N );

    printf("GTEST: Thread %d reports integer vectors declared.\n", omp_get_thread_num());
    fflush(stdout);

    //Integer vector 1 long to count data accesses, because scalar view bugs (previously)
    subscriber_vector_int_type counter( "DataAccesses", 1);

    Kokkos::Timer timer;

    counter(0) = 0;

    printf("GTEST: Thread %d reports counter successfully initialized to %d.\n", omp_get_thread_num(), counter(0));
    fflush(stdout);

    //Initialize y vector on host using parallel_for, increment a counter for data accesses.
    Kokkos::parallel_for( range_policy (0, N), KOKKOS_LAMBDA ( const int i) {
        y ( i ) = i;
        //printf("This is i from the parallel for %d \n", i);
        //fflush(stdout);
        Kokkos::atomic_increment(&counter(0));
        //printf("This is counter(0): %d\n", counter(0));
        //fflush(stdout);

    });

    Kokkos::fence();

    double time = timer.seconds();
    printf("GTEST: Thread %d reports counter is %d. It should be %d.\n", omp_get_thread_num(), counter(0), N);
    fflush(stdout);

    Kokkos::deep_copy(x, y);

    for ( int i = 0; i < N; i++) {
        ASSERT_EQ(x(i), i);
    }

    printf("GTEST: Thread %d reports test parallel_for completed. Data assignment was correct.\n", omp_get_thread_num());
    fflush(stdout);

    ASSERT_EQ(counter(0), N);

    printf("\n\n\n");
    fflush(stdout);
}

// gTest attempts to trigger all 3 executions generating different data.
// Requires non-multipe of 3 OMP threads to generate error.
// Should repeat user-specified number of times (in context file) and then abort.
TEST(TestResOpenMP, TestResilientForInsertError)
{
    std::cout << "KokkosResilient For OMP Thread Error" << std::endl;
    using range_policy = Kokkos::RangePolicy<ExecSpace2>;

  using subscriber_vector_int_type = Kokkos::View< int* , Kokkos::LayoutRight, MemSpace,
  Kokkos::Experimental::SubscribableViewHooks<
  KokkosResilience::ResilientDuplicatesSubscriber > >;
  using subscriber_vector_double_type = Kokkos::View< double* , Kokkos::LayoutRight, MemSpace,
  Kokkos::Experimental::SubscribableViewHooks<
  KokkosResilience::ResilientDuplicatesSubscriber > >;

  subscriber_vector_int_type counter ( "DataAccesses", 1);

  // Allocate y, x vectors.
  subscriber_vector_double_type y( "y", N );
  subscriber_vector_double_type x( "x", N );

  counter(0) = 0;

  printf("GTEST: Thread %d reports counter successfully initialized to %d.\n", omp_get_thread_num(), counter(0));
  fflush(stdout);

  ::testing::FLAGS_gtest_death_test_style = "threadsafe";

  // Assigning each y(i) threadId, should cause a failure in the resilient execution.
  EXPECT_DEATH(
  Kokkos::parallel_for( range_policy (0, N), KOKKOS_LAMBDA ( const int i) {
    y(i) = omp_get_thread_num();
    //std::cout << y(i) << std::endl;
    Kokkos::atomic_increment(&counter(0));
  });
,"Aborted in parallel_for, resilience majority voting failed because each execution obtained a differing value.");

  printf("\n\n\n");
  fflush(stdout);
}
//*/

//************
// MDrange test_for2 copy
//*************
namespace Test {
TEST(TestResOpenMP, TestMDrangeTest_for2) {

  std::cout << "Kokkos Normal MDRangePolicy Test" << std::endl;

  TestMDRange_2D<Kokkos::OpenMP>::test_for2(100, 100);
}

TEST(TestResOpenMP, TestResilientMDrangeTest_for2) {

  std::cout << "Resilient Kokkos Normal MDRangePolicy Test" << std::endl;

  TestMDRange_2D<ExecSpace2>::test_for2(100, 100);
}
}  // namespace Test

// */
/*********************************
********PARALLEL REDUCES**********
**********************************/
/*
// gTest runs parallel_reduce with non-resilient Kokkos. Should never fail.
TEST(TestResOpenMP, TestKokkosReduce) {

  std::cout << "KokkosReduce Test" << std::endl;

  // Non-resilient policy and views with no subscriber
  using range_policy2 = Kokkos::RangePolicy<Kokkos::OpenMP>;
  using ViewVectorType2 = Kokkos::View<double *, Kokkos::LayoutRight, Kokkos::HostSpace>;

  // Allocate y2, x2 vectors.
  ViewVectorType2 y2("y", N);
  ViewVectorType2 x2("x", N);

  double result = 0;
  double correct = (double) N;

  // Initialize y2 vector on host using parallel_for
  Kokkos::parallel_for(range_policy2(0, N), KOKKOS_LAMBDA(int i) {
    y2(i) = 1;
  });

  Kokkos::fence();

  Kokkos::deep_copy(x2, y2);

  Kokkos::Timer timer;

  // Perform vector dot product y2*x2 using parallel_reduce
  Kokkos::parallel_reduce("y2x2", N, KOKKOS_LAMBDA(int j, double &update) {
    update += y2(j) * x2(j);
  }, result);

  Kokkos::fence();
  double time = timer.seconds();

  printf("It took %f seconds to perform the Kokkos parallel_reduce.\n", time);
  fflush(stdout);
  printf("The correct dot product of two length-%d vectors of all ones is %d.\n", (int) N, (int) N);
  fflush(stdout);
  printf("The result from Kokkos parallel_reduce was %f.\n\n\n", result);
  fflush(stdout);

  ASSERT_EQ(result, correct);
}


// gTest runs parallel_reduce with resilient Kokkos. Expect same answer as last test.
TEST(TestResOpenMP, TestResilientReduce)
{

    std::cout << "KokkosResilient Reduce Doubles" << std::endl;
    // range policy with resilient execution space
    using range_policy = Kokkos::RangePolicy<ExecSpace>;

    // test vector types with the duplicating subscriber
    using subscriber_vector_double_type = Kokkos::View< double* , Kokkos::LayoutRight, MemSpace,
            Kokkos::Experimental::SubscribableViewHooks<
                    KokkosResilience::ResilientDuplicatesSubscriber > >;

    // Allocate y, x vectors.
    subscriber_vector_double_type y2( "y", N );
    subscriber_vector_double_type x2( "x", N );

    printf("GTEST: Thread %d reports resilient reduce vectors declared.\n", omp_get_thread_num());
    fflush(stdout);

    double result;
    double correct = (double) N;

    //Initialize y vector on host using parallel_for
    Kokkos::parallel_for( range_policy (0, N), KOKKOS_LAMBDA ( const int i) {
        y2( i ) = 1;
        //printf("This is i from the parallel for %d \n", i);
    });

    Kokkos::fence();

    Kokkos::deep_copy(x2, y2);

    Kokkos::Timer timer;

    // Perform vector dot product y2*x2 using parallel_reduce
    Kokkos::parallel_reduce("y2x2", N, KOKKOS_LAMBDA(int j, double &update) {
        update += y2( j ) * x2( j );
    }, result);

    Kokkos::fence();
    double time = timer.seconds();

    printf("It took %f seconds to perform the resilient parallel_reduce.\n", time);
    fflush(stdout);
    printf("The correct dot product of two length-%d vectors of all ones is %d.\n", (int) N, (int) N);
    fflush(stdout);
    printf("The result from resilient parallel_reduce was %f.\n\n\n", result);
    fflush(stdout);

    ASSERT_EQ(result, correct);
}
 */