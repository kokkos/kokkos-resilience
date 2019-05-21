#ifndef INC_RESILIENCE_TRACKER_HPP
#define INC_RESILIENCE_TRACKER_HPP

#include <vector>
#include <memory>

namespace KokkosResilience
{
  template< typename Holder >
  struct SerializationTracker
  {
    static bool enable_view_serialization;
    static std::vector< std::unique_ptr< Holder > > serialized_view_holders;
  };
  
  template< typename Holder >
  bool
    SerializationTracker< Holder >::enable_view_serialization = false;
  
  template< typename Holder >
  std::vector< std::unique_ptr< Holder > >
    SerializationTracker< Holder >::serialized_view_holders;
}

#endif  // INC_RESILIENCE_TRACKER_HPP
