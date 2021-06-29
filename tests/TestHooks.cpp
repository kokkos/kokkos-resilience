#include "TestCommon.hpp"
#include <resilience/AutomaticCheckpoint.hpp>
#include <resilience/ResilientRef.hpp>
#include <algorithm>
#include <Kokkos_View.hpp>

struct TestSubscriber;

static_assert( Kokkos::Experimental::is_hooks_policy< Kokkos::Experimental::SubscribableViewHooks< TestSubscriber > >::value, "Must be a hooks policy" );

using test_view_type = Kokkos::View< double **, Kokkos::Experimental::SubscribableViewHooks< TestSubscriber > >;

struct TestSubscriber
{
  static test_view_type *self_ptr;
  static const test_view_type *other_ptr;

  template< typename View >
  static void copy_constructed( View &self, const View &other )
  {
    self_ptr = &self;
    other_ptr = &other;
  }
};

test_view_type *TestSubscriber::self_ptr = nullptr;
const test_view_type *TestSubscriber::other_ptr = nullptr;

TEST(Subscriber, value)
{
  test_view_type testa;

  test_view_type testb( testa );
  std::cout << "Print from Nic's test, which will pass." << std::endl;
  EXPECT_EQ( TestSubscriber::self_ptr, &testb );
  EXPECT_EQ( TestSubscriber::other_ptr, &testa );
}