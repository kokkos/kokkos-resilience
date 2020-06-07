#ifndef INC_RESILIENCE_CONTEXT_HPP
#define INC_RESILIENCE_CONTEXT_HPP

#include <Kokkos_Macros.hpp>
#if defined(KOKKOS_ENABLE_HPX)
#include <hpx/config.hpp>
#endif
#include <string>
#include <utility>
#include <memory>
#include <functional>
#include <chrono>
#include "Config.hpp"
#include "Cref.hpp"
#include "CheckpointFilter.hpp"
#include <Kokkos_Core.hpp>
#include <Kokkos_ViewHooks.hpp>
#ifdef KR_ENABLE_MPI_BACKENDS
#include <mpi.h>
#endif

// Tracing support
#ifdef KR_ENABLE_TRACING
#include "util/Trace.hpp"
#endif

namespace KokkosResilience
{
  namespace detail
  {
  }

  class ContextBase
  {
  public:

    explicit ContextBase( Config cfg );

    virtual ~ContextBase() = default;

    virtual void register_hashes(const std::vector< std::unique_ptr< Kokkos::ViewHolderBase > > &views,
                                 const std::vector< Detail::CrefImpl > &crefs) = 0;
    virtual bool restart_available( const std::string &label, int version ) = 0;
    virtual void restart( const std::string &label, int version,
                          const std::vector< std::unique_ptr< Kokkos::ViewHolderBase > > &views ) = 0;
    virtual void checkpoint( const std::string &label, int version,
                             const std::vector< std::unique_ptr< Kokkos::ViewHolderBase > > &views ) = 0;

    virtual int latest_version( const std::string &label ) const noexcept = 0;

    virtual void reset() = 0;

    const std::function< bool( int ) > &default_filter() const noexcept { return m_default_filter; }

    Config &config() noexcept { return m_config; }
    const Config &config() const noexcept { return m_config; }

#ifdef KR_ENABLE_TRACING
    Util::detail::TraceStack  &trace() { return m_trace; };
#endif

  private:

    Config m_config;

    std::function< bool( int ) > m_default_filter;

#ifdef KR_ENABLE_TRACING
    Util::detail::TraceStack  m_trace;
#endif
  };

  std::unique_ptr< ContextBase > make_context( const std::string &config );
#ifdef KR_ENABLE_MPI_BACKENDS
  std::unique_ptr< ContextBase > make_context( MPI_Comm comm, const std::string &config );
#endif
#ifdef KR_ENABLE_STDFILE
  std::unique_ptr< ContextBase > make_context( const std::string &filename, const std::string &config );
#endif
}

#endif  // INC_RESILIENCE_CONTEXT_HPP
