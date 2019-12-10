#ifndef INC_RESILIENCE_AUTOMATICCHECKPOINT_HPP
#define INC_RESILIENCE_AUTOMATICCHECKPOINT_HPP

#include <string>
#include <vector>
#include <memory>
#include <ctime>
#include <iomanip>

#include <Kokkos_Core.hpp>
#include <Kokkos_ViewHooks.hpp>

#include "Cref.hpp"
#include "CheckpointFilter.hpp"

// Tracing support
#ifdef KR_ENABLE_TRACING
#include "util/Trace.hpp"
#include <sstream>
#endif

// Workaround for C++ < 17
#define KR_CHECKPOINT_THIS _kr_self = *this
#define KR_CHECKPOINT( x ) _kr_chk_##x = kr::check_ref< int >( x )

namespace KokkosResilience
{

  template< typename Context >
  int latest_version( Context &ctx, const std::string &label )
  {

    return ctx.latest_version( label );
  }

  namespace Detail
  {
    template< typename Context, typename F, typename FilterFunc >
    void checkpoint_impl( Context &ctx, const std::string &label, int iteration, F &&fun, FilterFunc &&filter )
    {
  #if defined( KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST )

      // Trace if enabled
  #ifdef KR_ENABLE_TRACING
      std::ostringstream oss;
      oss << "checkpoint_" << label;
      auto chk_trace = Util::begin_trace< Util::IterTimingTrace< std::string > >( ctx, oss.str(), iteration );

      auto overhead_trace = Util::begin_trace< Util::TimingTrace< std::string > >( ctx, "overhead" );
  #endif

      using fun_type = typename std::remove_reference< F >::type;

      if ( filter( iteration ) )
      {
        // Copy the functor, since if it has any views we can turn on view tracking
        std::vector< std::unique_ptr< Kokkos::ViewHolderBase > > views;

        // Don't do anything with const views since they can never be checkpointed in this context
        Kokkos::ViewHooks::set( [&views]( Kokkos::ViewHolderBase &view ) {
          views.emplace_back( view.clone() );
        }, []( Kokkos::ConstViewHolderBase & ) {} );

        std::vector< Detail::CrefImpl > crefs;
        Detail::Cref::check_ref_list = &crefs;

        fun_type f = fun;

        Detail::Cref::check_ref_list = nullptr;

        Kokkos::ViewHooks::clear();

  #ifdef KR_ENABLE_TRACING
        auto reg_hashes = Util::begin_trace< Util::TimingTrace< std::string > >( ctx, "register" );
  #endif
        // Register any views that haven't already been registered
        ctx.register_hashes( views, crefs );

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
          ctx.restart( label, iteration, views );
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
            auto ts = std::chrono::system_clock::to_time_t( std::chrono::system_clock::now() );
            std::cout << '[' << std::put_time( std::localtime( &ts ), "%c" ) << "] initiating checkpoint\n";
            ctx.checkpoint( label, iteration, views );
          }
        }
      } else
      {  // Iteration is filtered, just execute
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
  #else
  #ifdef KR_ENABLE_TRACING
      auto function_trace = Util::begin_trace< Util::TimingTrace< std::string > >( "function" );
  #endif
    fun();
  #ifdef KR_ENABLE_TRACING
      function_trace.end();
  #endif
  #endif
    }
  }

  template< typename Context, typename F, typename FilterFunc >
  void checkpoint( Context &ctx, const std::string &label, int iteration, F &&fun, FilterFunc &&filter )
  {
    Detail::checkpoint_impl( ctx, label, iteration, std::forward< F >( fun ), std::forward< FilterFunc >( filter ) );
  }

  template< typename Context, typename F >
  void checkpoint( Context &ctx, const std::string &label, int iteration, F &&fun )
  {
    Detail::checkpoint_impl( ctx, label, iteration, std::forward< F >( fun ), ctx.default_filter() );
  }

}

namespace kr = KokkosResilience;

#endif  // INC_RESILIENCE_AUTOMATICCHECKPOINT_HPP
