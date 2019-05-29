#ifndef INC_RESILIENCE_CHECKPOINT_HPP
#define INC_RESILIENCE_CHECKPOINT_HPP

#include <string>
#include <vector>
#include <memory>

#include <Kokkos_Core.hpp>
#include <Kokkos_ViewHooks.hpp>

namespace KokkosResilience
{
  namespace filter
  {
    struct default_filter
    {
      bool operator()( int ) const
      { return true; }
    };
    
    template< int Freq >
    struct nth_iteration_filter
    {
      bool operator()( int i ) const { return !( i % Freq ); }
    };
  }
  
  template< typename Context, typename F, typename FilterFunc = filter::default_filter >
  void checkpoint( Context &ctx, const std::string &label, int iteration, F &&fun, FilterFunc &&filter = filter::default_filter{} )
  {
#if defined( KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST )
    using fun_type = typename std::remove_reference< F >::type;
    
    // Copy the functor, since if it has any views we can turn on view tracking
    std::vector< std::unique_ptr< Kokkos::ViewHolderBase > > views;
    
    // Don't do anything with const views since they can never be checkpointed in this context
    Kokkos::ViewHooks::set( [&views]( Kokkos::ViewHolderBase &view ) {
      views.emplace_back( view.clone() );
    }, []( Kokkos::ConstViewHolderBase & ) {} );
    
    fun_type f = fun;
    
    Kokkos::ViewHooks::clear();
    
    if ( ctx.backend().restart_available( label, iteration ) )
    {
      // Load views with data
      ctx.backend().restart( label, iteration, views );
    } else {
      // Execute functor and checkpoint
      fun();
  
      if ( filter( iteration ) )
        ctx.backend().checkpoint( label, iteration, views );
    }
#else
    fun();
#endif
  }
}

#endif  // INC_RESILIENCE_CHECKPOINT_HPP
