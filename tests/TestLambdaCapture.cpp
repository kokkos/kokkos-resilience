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
#include <resilience/AutomaticCheckpoint.hpp>
#include <resilience/ResilientRef.hpp>
#include <algorithm>

template< typename F >
auto get_view_list( F &&_fun )
{
  std::vector< KokkosResilience::ViewHolder > views;
  KokkosResilience::DynamicViewHooks::copy_constructor_set.set_callback( [&views]( const KokkosResilience::ViewHolder &view ) {
    views.emplace_back( view );
  } );

  auto f = _fun;

  KokkosResilience::Detail::Cref::check_ref_list = nullptr;
  KokkosResilience::DynamicViewHooks::copy_constructor_set.reset();

  f();

  return views;
}

template< typename View >
bool capture_list_contains( const std::vector< KokkosResilience::ViewHolder > &_list, View &&_v )
{
  auto pos = std::find_if( _list.begin(), _list.end(), [&_v]( auto &&_hold ){ return _hold.data() == _v.data(); } );
  return pos != _list.end();
}

struct mixed_data
{
  mixed_data()
    : x( "test", 5 ), y( false )
  {}

  using view_type = Kokkos::View< double *, Kokkos::Experimental::SubscribableViewHooks< KokkosResilience::DynamicViewHooksSubscriber > >;
  view_type x;
  bool y;

  void work() { y = true; };
};

TEST(LambdaCapture, value)
{
  mixed_data dat;

  auto captures = get_view_list( [=]() mutable { dat.work(); } );

  EXPECT_FALSE( dat.y );
  EXPECT_TRUE( capture_list_contains( captures, dat.x ) );
}

TEST(LambdaCapture, reference)
{
  mixed_data dat;
  auto &ref = dat;

  auto captures = get_view_list( [&]() mutable { ref.work(); } );

  EXPECT_TRUE( dat.y );
  EXPECT_FALSE( capture_list_contains( captures, dat.x ) );
}

TEST(LambdaCapture, clone_holder)
{
  auto dat = mixed_data();

  auto holder = KokkosResilience::make_view_holder( dat.x );
  auto h2 = holder;

  EXPECT_EQ( holder.data(), dat.x.data() );
  EXPECT_EQ( holder.data(), h2.data() );
  EXPECT_EQ( h2.data(), dat.x.data() );
}

TEST(LambdaCapture, holder)
{
  auto dat = mixed_data();
  auto ref = KokkosResilience::Ref< mixed_data >( dat );

  auto captures = get_view_list( [=]() mutable { ref->work(); } );

  // Variable not used but we need to assign to it
  [[maybe_unused]] auto *vd = dat.x.data();

  EXPECT_TRUE( dat.y );
  EXPECT_TRUE( capture_list_contains( captures, dat.x ) );
}
