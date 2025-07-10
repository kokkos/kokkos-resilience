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
#ifndef INC_RESILIENCE_AUTOMATICCHECKPOINT_HPP
#define INC_RESILIENCE_AUTOMATICCHECKPOINT_HPP

#include <string>
#include <vector>
#include <memory>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <unordered_set>

#include <Kokkos_Core.hpp>
#include "view_hooks/ViewHolder.hpp"
#include "view_hooks/DynamicViewHooks.hpp"

#include "resilience/registration/Registration.hpp"
#include "resilience/registration/Specialized.hpp"
#include "resilience/context/CheckpointFilter.hpp"

// Tracing support
#ifdef KR_ENABLE_TRACING
#include "util/Trace.hpp"
#include <sstream>
#endif

#define KR_CHECKPOINT(x) KokkosResilience::RegistrationInfo(x, std::string(#x))

namespace KokkosResilience
{

  template< typename Context >
  int latest_version( Context &ctx, const std::string &label )
  {

    return ctx.latest_version( label );
  }

  namespace Detail
  {
    template<typename RegionFunc, typename... T>
    void autodetect_members(
      ContextBase& ctx, RegionFunc&& fun,
      std::unordered_set<Registration>& members
    ) {
        //Enable ViewHolder copy constructor hooks to register the views
        KokkosResilience::DynamicViewHooks::copy_constructor_set.set_callback(
          [&ctx, &members](const KokkosResilience::ViewHolder &view) {
            members.insert(Registration(ctx, view));
          }
        );
          
        //Copy the lambda/functor to trigger copy-constructor hooks
        using FuncType = typename std::remove_reference<RegionFunc>::type;
        [[maybe_unused]] FuncType f = fun;

        //Disable ViewHolder hook
        KokkosResilience::DynamicViewHooks::copy_constructor_set.reset();
    }

    template< typename RegionFunc, typename FilterFunc, typename... T>
    void checkpoint_impl(
      ContextBase &ctx, const std::string &label, int iteration,
      RegionFunc &&fun, FilterFunc &&filter,
      RegistrationInfo<T>... explicit_members
    ) {
      // Trace if enabled
  #ifdef KR_ENABLE_TRACING
      std::ostringstream oss;
      oss << "checkpoint_" << label;
      auto chk_trace = Util::begin_trace< Util::IterTimingTrace< std::string > >( ctx, oss.str(), iteration );

      auto overhead_trace = Util::begin_trace< Util::TimingTrace< std::string > >( ctx, "overhead" );
  #endif

      if ( filter( iteration ) )
      {
        //Gather up explicitly requested members and any autodetectable.
        std::unordered_set<Registration> members = { Registration(ctx, explicit_members)... };
        autodetect_members(ctx, fun, members);

  #ifdef KR_ENABLE_TRACING
        auto reg_hashes = Util::begin_trace< Util::TimingTrace< std::string > >( ctx, "register" );
  #endif
        // Register any views that haven't already been registered
        ctx.register_hashes( members );

  #ifdef KR_ENABLE_TRACING
        reg_hashes.end();
        auto check_restart = Util::begin_trace< Util::TimingTrace< std::string > >( ctx, "check" );
  #endif

        bool restart_available = ctx.restart_available( label, iteration );
  #ifdef KR_ENABLE_TRACING
        check_restart.end();
        overhead_trace.end();
  #endif

        if ( restart_available )
        {
          // Load views with data
  #ifdef KR_ENABLE_TRACING
          auto restart_trace = Util::begin_trace< Util::TimingTrace< std::string > >( ctx, "restart" );
  #endif
          ctx.restart( label, iteration, members );
        } else
        {
          // Execute functor and checkpoint
  #ifdef KR_ENABLE_TRACING
          auto function_trace = Util::begin_trace< Util::TimingTrace< std::string > >( ctx, "function" );
  #endif
          fun();
  #ifdef KR_ENABLE_TRACING
          Kokkos::fence();  // Get accurate measurements for function_trace end
          function_trace.end();
  #endif

          {
  #ifdef KR_ENABLE_TRACING
            auto write_trace = Util::begin_trace< Util::TimingTrace< std::string > >( ctx, "checkpoint" );
  #endif
            Kokkos::fence();
            auto ts = std::chrono::system_clock::to_time_t( std::chrono::system_clock::now() );
            std::cout << '[' << std::put_time( std::localtime( &ts ), "%c" ) << "] initiating checkpoint\n";
            ctx.checkpoint( label, iteration, members );
          }
        }
      } else {  // Iteration is filtered, just execute
  #ifdef KR_ENABLE_TRACING
        overhead_trace.end();
        auto function_trace = Util::begin_trace< Util::TimingTrace< std::string > >( ctx, "function" );
  #endif
        fun();
  #ifdef KR_ENABLE_TRACING
        Kokkos::fence();  // Get accurate measurements for function_trace end
        function_trace.end();
  #endif
      }
    }
  }

  template<typename FilterFunc>
  constexpr bool is_filter_v = std::is_same_v<
    std::invoke_result<FilterFunc, int>, 
    bool
  >;

  template< typename Context, typename F, typename FilterFunc, typename... T, std::enable_if_t<is_filter_v<FilterFunc>>* = nullptr>
  void checkpoint( Context &ctx, const std::string &label, int iteration, F &&fun, FilterFunc &&filter, RegistrationInfo<T>... explicit_members)
  {
    Detail::checkpoint_impl( ctx, label, iteration, std::forward< F >( fun ), std::forward< FilterFunc >( filter ), explicit_members...);
  }

  template< typename Context, typename F, typename... T>
  void checkpoint( Context &ctx, const std::string &label, int iteration, F &&fun, RegistrationInfo<T>... explicit_members)
  {
    Detail::checkpoint_impl( ctx, label, iteration, std::forward< F >( fun ), ctx.default_filter(), explicit_members... );
  }

}

namespace kr = KokkosResilience;

#endif  // INC_RESILIENCE_AUTOMATICCHECKPOINT_HPP
