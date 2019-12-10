#ifndef INC_RESILIENCE_CHECKPOINT_FILTER_HPP
#define INC_RESILIENCE_CHECKPOINT_FILTER_HPP

#include <chrono>

namespace KokkosResilience
{
  namespace Filter
  {
    struct DefaultFilter
    {
      bool operator()( int ) const
      { return true; }
    };

    template< int Freq >
    struct StaticNthIterationFilter
    {
      bool operator()( int i ) const { return !( i % Freq ); }
    };

    struct NthIterationFilter
    {
      explicit NthIterationFilter( int freq )
        : frequency( freq )
      {}

      bool operator()( int i ) const
      {
        return !( i % frequency );
      }

      int frequency;
    };

    struct TimeFilter
    {
      using clock_type = std::chrono::steady_clock;
      using duration_type = std::chrono::steady_clock::duration;
      using time_point_type = std::chrono::steady_clock::time_point;

      template< typename Rep, typename Period >
      explicit TimeFilter( std::chrono::duration< Rep, Period > duration )
          : checkpoint_interval( std::chrono::duration_cast< duration_type >( duration ) ),
            start( clock_type::now() )
      {
      }

      bool operator()( int ) const
      {
        auto now = clock_type::now();
        bool ret = ( now - start ) > checkpoint_interval;
        if ( ret )
          start = now;

        return ret;
      }

      mutable time_point_type start;
      duration_type checkpoint_interval;
    };
  }
}

#endif  // INC_RESILIENCE_CHECKPOINT_FILTER_HPP
