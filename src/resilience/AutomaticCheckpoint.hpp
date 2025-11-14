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

#include "util/Trace.hpp"

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
      KR_TRACE_ITERATION(ctx, label, iteration);

      KR_TRACE(overhead, ctx);
      std::optional<std::unordered_set<Registration>> members;
      bool is_checkpoint_iter = filter(iteration);
      bool restart_available = false;
      if (is_checkpoint_iter) {
        KR_TRACE(registration, ctx);
          // Weird initialization for case with no explicit members to ensure we
          // are initializing a set, not an empty optional
          members.emplace(std::move(
              std::unordered_set<Registration> { Registration(ctx, explicit_members)... }
          ));
          autodetect_members(ctx, fun, *members);
          ctx.register_hashes(*members);
        KR_TRACE_END(registration);

        KR_TRACE(check_restart, ctx);
         restart_available = ctx.restart_available( label, iteration );
        KR_TRACE_END(check_restart);
      }
      KR_TRACE_END(overhead);

      if (restart_available) {
        KR_TRACE(restart, ctx);
          ctx.restart( label, iteration, *members );
        KR_TRACE_END(restart);
        return;
      }

      KR_TRACE(function, ctx);
        fun();
      KR_TRACE_END_FENCE(function);

      if (is_checkpoint_iter) {
        KR_TRACE(checkpoint, ctx);
          Kokkos::fence();
          auto ts = std::chrono::system_clock::to_time_t( std::chrono::system_clock::now() );
          std::cout << '[' << std::put_time( std::localtime( &ts ), "%c" ) << "] initiating checkpoint\n";
          ctx.checkpoint( label, iteration, *members );
        KR_TRACE_END(checkpoint);
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
