#ifndef INC_RESILIENCE_AUTOMATICCHECKPOINT_HPP
#define INC_RESILIENCE_AUTOMATICCHECKPOINT_HPP

#include <string>
#include <vector>
#include <memory>

#include <Kokkos_Core.hpp>
#include <Kokkos_ViewHooks.hpp>

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
  namespace filter
  {
    struct default_filter
    {
      bool operator()( int ) const
      { return true; }
    };
    
    template< int Freq >
    struct nth_iteration_filter
    {
      bool operator()( int i ) const { return !( i % Freq ); }
    };
  }

  template< typename Context >
  int latest_version(Context &ctx, const std::string &label) {

    return ctx.backend().latest_version(label);
  }
  
  namespace detail
  {
    struct cref_impl
    {
      cref_impl( void *p, std::size_t s, std::size_t n )
        : ptr( p ), sz( s ), num( n )
      {}
      
      void        *ptr;
      std::size_t sz;
      std::size_t num;
    };
    
    struct cref : public cref_impl
    {
      using cref_impl::cref_impl;
      cref( const cref &_other )
        : cref_impl( _other.ptr, _other.sz, _other.num )
      {
        if ( check_ref_list )
          check_ref_list->emplace_back( ptr, sz, num );
      }
      
      static std::vector< cref_impl > *check_ref_list;
    };
  }
  
  template< typename T >
  auto check_ref( T &_t )
  {
    return detail::cref{ reinterpret_cast< void * >( &_t ), sizeof( T ), 1 };
  }
  
  template< typename Context, typename F, typename FilterFunc = filter::default_filter >
  void checkpoint( Context &ctx, const std::string &label, int iteration, F &&fun, FilterFunc &&filter = filter::default_filter{} )
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
        views.emplace_back( view.clone());
      }, []( Kokkos::ConstViewHolderBase & ) {} );
  
      std::vector< detail::cref_impl > crefs;
      detail::cref::check_ref_list = &crefs;
  
      fun_type f = fun;
  
      detail::cref::check_ref_list = nullptr;
  
      Kokkos::ViewHooks::clear();

#ifdef KR_ENABLE_TRACING
      auto reg_hashes = Util::begin_trace< Util::TimingTrace< std::string > >( ctx, "register" );
#endif
      // Register any views that haven't already been registered
      ctx.backend().register_hashes( views, crefs );

#ifdef KR_ENABLE_TRACING
      reg_hashes.end();
      auto check_restart = Util::begin_trace< Util::TimingTrace< std::string > >( ctx, "check" );
#endif
  
      bool restart_available = ctx.backend().restart_available( label, iteration );
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
        ctx.backend().restart( label, iteration, views );
      }
      else
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
          ctx.backend().checkpoint( label, iteration, views );
        }
      }
    } else {  // Iteration is filtered, just execute
#ifdef KR_ENABLE_TRACING
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

namespace kr = KokkosResilience;

#endif  // INC_RESILIENCE_AUTOMATICCHECKPOINT_HPP
