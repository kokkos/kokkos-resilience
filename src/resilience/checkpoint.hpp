#ifndef INC_RESILIENCE_CHECKPOINT_HPP
#define INC_RESILIENCE_CHECKPOINT_HPP

#include <string>
#include <vector>
#include <memory>

#include <Kokkos_Core.hpp>
#include <Kokkos_ViewHooks.hpp>

namespace KokkosResilience
{
  template< typename F, typename CheckpointBackend >
  void checkpoint( const std::string &label, int iteration, F &&fun, CheckpointBackend &check )
  {
#if defined( KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST )
    using fun_type = typename std::remove_reference< F >::type;
    
    // Copy the functor, since if it has any views we can turn on view tracking
    std::vector< std::unique_ptr< Kokkos::ViewHolderBase > > views;
    Kokkos::ViewHooks::set( [&views]( Kokkos::ViewHolderBase &view ) {
      views.emplace_back( view.clone() );
    } );
    
    fun_type f = fun;
    
    Kokkos::ViewHooks::clear();
    
    if ( check.restart_available( label, iteration ) )
    {
      // Load views with data
      check.restart( label, iteration, views );
    } else {
      // Execute functor and checkpoint
      fun();
      check.checkpoint( label, iteration, views );
    }
#else
    fun();
#endif
  }
}

#endif  // INC_RESILIENCE_CHECKPOINT_HPP
