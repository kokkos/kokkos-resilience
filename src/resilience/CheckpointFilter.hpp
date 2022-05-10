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

      template <typename Rep, typename Period>
      explicit TimeFilter(std::chrono::duration<Rep, Period> duration)
          : start(clock_type::now()),
            checkpoint_interval(
                std::chrono::duration_cast<duration_type>(duration)) {}

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
