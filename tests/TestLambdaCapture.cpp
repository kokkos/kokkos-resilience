#include "TestCommon.hpp"
#include <resilience/AutomaticCheckpoint.hpp>
#include <resilience/ResilientRef.hpp>
#include <algorithm>

template< typename F >
auto get_view_list( F &&_fun )
{
  std::vector< std::unique_ptr< Kokkos::ViewHolderBase > > views;
  Kokkos::ViewHooks::set( [&views]( Kokkos::ViewHolderBase &view ) {
    views.emplace_back( view.clone() );
  }, []( Kokkos::ConstViewHolderBase & ) {} );

  auto f = _fun;

  KokkosResilience::Detail::Cref::check_ref_list = nullptr;
  Kokkos::ViewHooks::clear();

  f();

  return views;
}

template< typename View >
bool capture_list_contains( const std::vector< std::unique_ptr< Kokkos::ViewHolderBase > > &_list, View &&_v )
{
  auto pos = std::find_if( _list.begin(), _list.end(), [&_v]( auto &&_hold ){ return _hold->data() == _v.data(); } );
  return pos != _list.end();
}

struct mixed_data
{
  mixed_data()
    : x( "test", 5 ), y( false )
  {}

  Kokkos::View< double * > x;
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

  auto holder = Kokkos::ViewHolder< decltype( dat.x ) >( dat.x );
  auto *h2 = holder.clone();

  EXPECT_EQ( holder.data(), dat.x.data() );
  EXPECT_EQ( holder.data(), h2->data() );
  EXPECT_EQ( h2->data(), dat.x.data() );
}

TEST(LambdaCapture, holder)
{
  auto dat = mixed_data();
  auto ref = KokkosResilience::Ref< mixed_data >( dat );

  auto captures = get_view_list( [=]() mutable { ref->work(); } );

  auto *vd = dat.x.data();

  EXPECT_TRUE( dat.y );
  EXPECT_TRUE( capture_list_contains( captures, dat.x ) );
}