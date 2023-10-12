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

#include <Kokkos_Core.hpp>
#include "view_hooks/ViewHolder.hpp"
#include "view_hooks/DynamicViewHooks.hpp"
#include "registration/ViewHolder.hpp"

#include "context/ContextBase.hpp"

#include "CheckpointFilter.hpp"

// Tracing support
#include "util/Trace.hpp"
#include <sstream>

#define KR_CHECKPOINT(x) KokkosResilience::Detail::RegInfo(x, std::string(#x))
//#define KR_REGISTER( context, x ) _kr_chk_##x = kr::check_ref<decltype( context )>( #x, x )

namespace KokkosResilience
{
  template< typename Context >
  int latest_version( Context &ctx, const std::string &label )
  {
    return ctx.latest_version( label );
  }

  template<typename... Traits, typename F, typename... T>
  void ContextBase::detect_and_register(F&& fun, Detail::RegInfo<T>... explicit_members){
    using namespace Util;
#ifdef KR_ENABLE_TRACING
    auto reg_hashes = begin_trace<TimingTrace>( *this, "register" );
#endif

    //Gather up the explicitly-listed members.
    (register_to_active<Traits...>(explicit_members), ...);

    //Enable ViewHolder copy constructor hooks to register the views
    KokkosResilience::DynamicViewHooks::copy_constructor_set.set_callback(
      [ctx = this](const KokkosResilience::ViewHolder &view) {
        ctx->register_to_active<Traits...>(view);
      }
    );

    //Copy the lambda/functor to trigger copy-constructor hooks
    using FuncType = typename std::remove_reference<F>::type;
    [[maybe_unused]] FuncType f = fun;

    //Disable ViewHolder hook
    KokkosResilience::DynamicViewHooks::copy_constructor_set.reset();

#ifdef KR_ENABLE_TRACING
    reg_hashes.end();
#endif
  }

  template<typename... Traits, typename RegionFunc, typename FilterFunc, typename... T>
  void ContextBase::run(const std::string &label, int iteration, RegionFunc&& fun,
                        FilterFunc&& filter, Detail::RegInfo<T>&... explicit_members)
  {
    using namespace Util;
#ifdef KR_ENABLE_TRACING
    //Only build iteration label if tracing.
    std::ostringstream oss;
    oss << "checkpoint_" << label;
    auto chk_trace = begin_trace<IterTimingTrace>(*this, oss.str(), iteration);
#endif
    auto overhead_trace = begin_trace<TimingTrace>( *this, "overhead" );

    active_context = this;

    //Figure out how we should be handling this
    bool recover_region = false, checkpoint_region = false;
   
    auto parent_region = active_region;
    auto parent_context = active_context;
    auto* parent_filter = active_filter;

    if(last_region && last_region.label() == label) {
      active_region = last_region;
    } else {
      auto info = regions.insert({std::string(label), std::unordered_set<Registration>()});
      active_region = info.first;
    }
    std::function< bool(int) > m_filter = filter;
    active_filter = &m_filter;


    if(filter(iteration)){
      //Make sure the data members are registered to the context for this region.
      detect_and_register<Traits...>(std::forward<RegionFunc>(fun), explicit_members...);

      auto check_restart = begin_trace<TimingTrace>( *this, "check" );
      recover_region = this->restart_available(label, iteration);
      check_restart.end();

      checkpoint_region = !recover_region;
    }
    overhead_trace.end();


    if(recover_region){
      auto restart_trace = begin_trace<TimingTrace>( *this, "restart" );
      this->restart(active_region.label(), iteration, active_region.members());
    } else {
      enter_region(active_region, iteration);
      auto function_trace = begin_trace<TimingTrace>( *this, "function" );
      fun();
      function_trace.end();
      exit_region(active_region, iteration);

      if(checkpoint_region){
          auto write_trace = begin_trace<TimingTrace>( *this, "checkpoint" );
          auto ts = std::chrono::system_clock::to_time_t( std::chrono::system_clock::now() );
          if(m_pid == 0) std::cout << '[' << std::put_time( std::localtime( &ts ), "%c" ) << "] initiating checkpoint of iteration " << iteration << "\n";
          this->checkpoint(active_region.label(), iteration, active_region.members());
          write_trace.end();
      }
    }


    //Region no longer active
    last_region = active_region;
    active_region = parent_region;
    active_context = parent_context;
    active_filter = parent_filter;
  }

  //RegionFunc = std::function<void()>;
  //FilterFunc = std::function<bool(int)>;

  template<typename... Traits, typename T>
  bool register_if_active(T& member, std::string label){
    if(ContextBase::active_context != nullptr){
      return ContextBase::active_context->register_if_active<Traits...>(member, label);
    }
    return false;
  }
  
  template<typename... Traits, typename T>
  bool deregister_if_active(T& member, std::string label){
    if(ContextBase::active_context != nullptr){
      return ContextBase::active_context->deregister_if_active<Traits...>(member, label);
    }
    return false;
  }

  template<typename... Traits, typename F, typename FilterFunc, typename... T>
  void checkpoint( ContextBase& ctx, const std::string &label, int iteration, F &&fun, FilterFunc &&filter, Detail::RegInfo<T>... explicit_members )
  {
    ctx.run<Traits...>(label, iteration, std::forward< F >( fun ), std::forward< FilterFunc >( filter ), explicit_members...);
  }

  template< typename... Traits, typename F, typename... T>
  void checkpoint( ContextBase& ctx, const std::string &label, int iteration, F&& fun, Detail::RegInfo<T>... explicit_members )
  {
    ctx.run<Traits...>(label, iteration, std::forward< F >( fun ), ctx.default_filter(), explicit_members... );
  }

}

namespace kr = KokkosResilience;

#endif  // INC_RESILIENCE_AUTOMATICCHECKPOINT_HPP
