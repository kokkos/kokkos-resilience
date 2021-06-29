#include "TestCommon.hpp"
#include <resilience/AutomaticCheckpoint.hpp>
#include <resilience/ResilientRef.hpp>
#include <algorithm>
#include <Kokkos_View.hpp>
#include <Kokkos_ViewHolder.hpp>

#include <resilience/openMP/OpenMPResSubscriber.hpp>

// Do all views used in resilient execution space now need to be declared with a subscriber? YES

using test_view_type = Kokkos::View< double **, Kokkos::Experimental::SubscribableViewHooks< TestIISubscriber > >;

test_view_type *TestIISubscriber::self_ptr = nullptr;
const test_view_type *TestIISubscriber::other_ptr = nullptr;

TEST(TestIISubscriber, value)
{

std::cout << "This is the moved subscriber test. It compiled." << std::endl;

// Where does the subscriber come in the declaration list? Check to get data in.
test_view_type testa;
test_view_type testb( testa );

EXPECT_EQ( TestIISubscriber::self_ptr, &testb );
EXPECT_EQ( TestIISubscriber::other_ptr, &testa );

std::cout << "This is the moved subscriber test. It ran correctly." << std::endl;

}