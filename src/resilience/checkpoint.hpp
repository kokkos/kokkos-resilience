#ifndef INC_RESILIENCE_CHECKPOINT_HPP
#define INC_RESILIENCE_CHECKPOINT_HPP

#include <string>
#include <vector>
#include <memory>

#include "util/view_holder.hpp"
#include "tracker.hpp"

namespace KokkosResilience
{
  template< typename F, typename CheckpointBackend >
  void checkpoint( const std::string &label, int iteration, F &&fun, CheckpointBackend &check )
  {
    using fun_type = typename std::remove_reference< F >::type;
    
    // Copy the functor, since if it has any views we can turn on view tracking
    SerializationTracker< ViewHolderBase >::enable_view_serialization = true;
    SerializationTracker< ViewHolderBase >::serialized_view_holders.clear();
    fun_type f = fun;
    SerializationTracker< ViewHolderBase >::enable_view_serialization = false;
    
    std::vector< std::unique_ptr< ViewHolderBase > > views;
    std::swap( views, SerializationTracker< ViewHolderBase >::serialized_view_holders );
    
    if ( check.restart_available( label, iteration ) )
    {
      // Load views with data
      check.restart( label, iteration, views );
    } else {
      // Execute functor and checkpoint
      fun();
      check.checkpoint( label, iteration, views );
    }
  }
}

#endif  // INC_RESILIENCE_CHECKPOINT_HPP
