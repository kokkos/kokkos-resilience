#include "TestCommon.hpp"
#include <resilience/view_hooks/ViewHolder.hpp>
#include <resilience/view_hooks/DynamicViewHooks.hpp>

template< typename Device >
class TestDynamicViewHooks : public ::testing::Test
{
 public:

  using device = Device;

  using test_view_type =
      Kokkos::View<double **,
                   Kokkos::Experimental::SubscribableViewHooks<
                       KokkosResilience::DynamicViewHooksSubscriber>,
                   device >;
  using const_test_view_type =
      Kokkos::View<const double **,
                   Kokkos::Experimental::SubscribableViewHooks<
                       KokkosResilience::DynamicViewHooksSubscriber>,
                   device >;
};


TYPED_TEST_SUITE( TestDynamicViewHooks, enabled_exec_spaces );

TYPED_TEST( TestDynamicViewHooks, TestDynamicViewHooksCopyConstruct )
{
  using test_view_type = typename TestFixture::test_view_type;
  using const_test_view_type = typename TestFixture::const_test_view_type;

  KokkosResilience::ViewHolder holder;
  KokkosResilience::ConstViewHolder const_holder;

  KokkosResilience::DynamicViewHooks::reset();

  KokkosResilience::DynamicViewHooks::copy_constructor_set.set_callback(
      [&holder](const KokkosResilience::ViewHolder &vh) mutable {
        holder = vh;
      });

  KokkosResilience::DynamicViewHooks::copy_constructor_set
      .set_const_callback(
          [&const_holder](
              const KokkosResilience::ConstViewHolder &vh) mutable {
            const_holder = vh;
          });

  test_view_type testa("testa", 10, 10);
  test_view_type testb(testa);
  EXPECT_EQ(testa.data(), holder.data());
  EXPECT_EQ(const_holder.data(), nullptr);
  const_test_view_type testa_const(
      testa);  // Won't trigger the callback since this is not a copy
               // constructor call
  const_test_view_type testb_const(testa_const);
  EXPECT_EQ(testa_const.data(), const_holder.data());
}

TYPED_TEST( TestDynamicViewHooks, TestDynamicViewHooksMoveConstruct )
{
  using test_view_type       = typename TestFixture::test_view_type;
  using const_test_view_type = typename TestFixture::const_test_view_type;

  KokkosResilience::ViewHolder holder;
  KokkosResilience::ConstViewHolder const_holder;

  KokkosResilience::DynamicViewHooks::reset();

  KokkosResilience::DynamicViewHooks::move_constructor_set.set_callback(
      [&holder](const KokkosResilience::ViewHolder &vh) mutable {
        holder = vh;
      });

  KokkosResilience::DynamicViewHooks::move_constructor_set
      .set_const_callback(
          [&const_holder](
              const KokkosResilience::ConstViewHolder &vh) mutable {
            const_holder = vh;
          });

  test_view_type testa("testa", 10, 10);
  void *cmp = testa.data();
  test_view_type testb(std::move(testa));
  EXPECT_EQ(cmp, holder.data());
  EXPECT_EQ(const_holder.data(), nullptr);
  const_test_view_type testa_const(
      testb);  // Won't trigger the callback since this is not a copy
               // constructor call
  const_test_view_type testb_const(std::move(testa_const));
  EXPECT_EQ(cmp, const_holder.data());
}

TYPED_TEST( TestDynamicViewHooks, TestDynamicViewHooksCopyAssign )
{
  using test_view_type       = typename TestFixture::test_view_type;
  using const_test_view_type = typename TestFixture::const_test_view_type;
  KokkosResilience::ViewHolder holder;
  KokkosResilience::ConstViewHolder const_holder;

  KokkosResilience::DynamicViewHooks::reset();

  KokkosResilience::DynamicViewHooks::copy_assignment_set.set_callback(
      [&holder](const KokkosResilience::ViewHolder &vh) mutable {
        holder = vh;
      });

  KokkosResilience::DynamicViewHooks::copy_assignment_set
      .set_const_callback(
          [&const_holder](
              const KokkosResilience::ConstViewHolder &vh) mutable {
            const_holder = vh;
          });

  test_view_type testa("testa", 10, 10);
  test_view_type testb;
  testb = testa;
  EXPECT_EQ(testa.data(), holder.data());
  EXPECT_EQ(const_holder.data(), nullptr);
  const_test_view_type testa_const(
      testa);  // Won't trigger the callback since this is not a copy
               // constructor call
  const_test_view_type testb_const;
  testb_const = testa_const;
  EXPECT_EQ(testa_const.data(), const_holder.data());
}

TYPED_TEST( TestDynamicViewHooks, TestDynamicViewHooksMoveAssign )
{
  using test_view_type       = typename TestFixture::test_view_type;
  using const_test_view_type = typename TestFixture::const_test_view_type;

  KokkosResilience::ViewHolder holder;
  KokkosResilience::ConstViewHolder const_holder;

  KokkosResilience::DynamicViewHooks::reset();

  KokkosResilience::DynamicViewHooks::move_assignment_set.set_callback(
      [&holder](const KokkosResilience::ViewHolder &vh) mutable {
        holder = vh;
      });

  KokkosResilience::DynamicViewHooks::move_assignment_set
      .set_const_callback(
          [&const_holder](
              const KokkosResilience::ConstViewHolder &vh) mutable {
            const_holder = vh;
          });

  test_view_type testa("testa", 10, 10);
  void *cmp = testa.data();
  test_view_type testb;
  testb = std::move(testa);
  EXPECT_EQ(cmp, holder.data());
  EXPECT_EQ(const_holder.data(), nullptr);
  const_test_view_type testa_const(
      testa);  // Won't trigger the callback since this is not a copy
               // constructor call
  const_test_view_type testb_const;
  testb_const = std::move(testa_const);
  EXPECT_EQ(cmp, const_holder.data());
}
