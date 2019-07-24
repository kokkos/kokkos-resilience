#ifndef INC_RESILIENCE_UTIL_TIMER_HPP
#define INC_RESILIENCE_UTIL_TIMER_HPP

#include <chrono>

namespace KokkosResilience
{
  namespace Util
  {
    class Timer
    {
    public:
      
      using clock_type = std::chrono::high_resolution_clock;
      using duration_type = clock_type::duration;
      using time_type = std::chrono::time_point< clock_type >;
      
      explicit Timer( bool start_timer = false )
      {
        if ( start_timer )
          start();
      }
    
      void start() noexcept { m_start = clock_type::now(); }
      duration_type time() const noexcept { return clock_type::now() - m_start; }
      
    private:
      
      time_type m_start;
    };
  }
}

#endif  // INC_RESILIENCE_UTIL_TIMER_HPP
