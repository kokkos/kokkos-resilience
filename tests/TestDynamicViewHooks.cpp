/*
 *
 *                        Kokkos v. 3.0
 *       Copyright (2020) National Technology & Engineering
 *               Solutions of Sandia, LLC (NTESS).
 *
 * Under the terms of Contract DE-NA0003525 with NTESS,
 * the U.S. Government retains certain rights in this software.
 *
 * Kokkos is licensed under 3-clause BSD terms of use:
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 * 1. Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the Corporation nor the names of the
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Questions? Contact Christian R. Trott (crtrott@sandia.gov)
 */
#include "TestCommon.hpp"
#include <resilience/view_hooks/ViewHolder.hpp>
#include <resilience/view_hooks/DynamicViewHooks.hpp>

/*
 * For each hook -- copy construct, move construct, copy assign, and move
 * assign, test the capture of both const and non-const types.
 */
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

  // Trigger the non-const copy-construct callback
  test_view_type testb(testa);
  EXPECT_EQ(testa.data(), holder.data());
  EXPECT_EQ(const_holder.data(), nullptr);
  const_test_view_type testa_const(
      testa);  // Won't trigger the callback since this is not a copy
               // constructor call

  // Trigger the const copy-construct callback
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

  // Trigger the non-const move-construct callback
  test_view_type testb(std::move(testa));
  EXPECT_EQ(cmp, holder.data());
  EXPECT_EQ(const_holder.data(), nullptr);
  const_test_view_type testa_const(
      testb);  // Won't trigger the callback since this is not a copy
               // constructor call

  // Trigger the const move-construct callback
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

  // Trigger the non-const copy assign callback
  testb = testa;
  EXPECT_EQ(testa.data(), holder.data());
  EXPECT_EQ(const_holder.data(), nullptr);
  const_test_view_type testa_const(
      testa);  // Won't trigger the callback since this is not a copy
               // constructor call
  const_test_view_type testb_const;

  // Trigger the const copy assign callback
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

  // Trigger the non-const move assign callback
  testb = std::move(testa);
  EXPECT_EQ(cmp, holder.data());
  EXPECT_EQ(const_holder.data(), nullptr);
  const_test_view_type testa_const(
      testa);  // Won't trigger the callback since this is not a copy
               // constructor call
  const_test_view_type testb_const;

  // Trigger the const move assign callback
  testb_const = std::move(testa_const);
  EXPECT_EQ(cmp, const_holder.data());
}
