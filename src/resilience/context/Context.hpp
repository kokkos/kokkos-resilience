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
#include "resilience/config/Config.hpp"
#include "resilience/Cref.hpp"
#include "CheckpointFilter.hpp"
#include <Kokkos_Core.hpp>
#include "resilience/view_hooks/ViewHolder.hpp"
#ifdef KR_ENABLE_MPI_CONTEXT
#include <mpi.h>
#endif

// Tracing support
#ifdef KR_ENABLE_TRACING
#include "resilience/util/Trace.hpp"
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

    virtual void register_hashes(const std::vector< KokkosResilience::ViewHolder > &views,
                                 const std::vector< Detail::CrefImpl > &crefs) = 0;
    virtual bool restart_available( const std::string &label, int version ) = 0;
    virtual void restart( const std::string &label, int version,
                          const std::vector< KokkosResilience::ViewHolder > &views ) = 0;
    virtual void checkpoint( const std::string &label, int version,
                             const std::vector< KokkosResilience::ViewHolder > &views ) = 0;

    virtual int latest_version( const std::string &label ) const noexcept = 0;
    virtual void register_alias( const std::string &original, const std::string &alias ) = 0;

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
  std::unique_ptr< ContextBase > make_context( const Config &config );
#ifdef KR_ENABLE_MPI_CONTEXT
  std::unique_ptr< ContextBase > make_context( MPI_Comm comm, const std::string &cfg );
  std::unique_ptr< ContextBase > make_context( MPI_Comm comm, const Config &cfg );
#endif
}

#endif  // INC_RESILIENCE_CONTEXT_HPP
