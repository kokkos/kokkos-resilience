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
#include <unordered_set>
#include "resilience/config/Config.hpp"
#include "resilience/registration/Registration.hpp"
#include "CheckpointFilter.hpp"
#include <Kokkos_Core.hpp>
#ifdef KR_ENABLE_MPI_CONTEXT
#include <mpi.h>
#endif

// Tracing support
#ifdef KR_ENABLE_TRACING
#include "resilience/util/Trace.hpp"
#endif

#include "resilience/backend/Backend.hpp"

namespace KokkosResilience
{
  class ContextBase
  {
  public:
    using FilterFunc = std::function<bool(int)>;
    using Members = std::unordered_set<Registration>;

    //Lets us use map<string,Members> elements with sane names
    struct Region
    {
      Region(std::pair<const std::string, Members>& map_elem)
        : label(map_elem.first), members(map_elem.second) { };
      const std::string& label;
      Members& members;
    };

    ContextBase( Config cfg, int proc_id );
    virtual ~ContextBase() = default;

    virtual bool restart_available(const std::string& label, int version) = 0;
    virtual void restart(
      const std::string& label, int version, Members& members
    ) = 0;
    virtual void checkpoint(
      const std::string& label, int version, Members& members
    ) = 0;
    virtual int latest_version( const std::string& label ) const noexcept = 0;

    void reset();
    virtual void reset_impl() = 0;
   
    virtual void register_alias(
      const std::string& original, const std::string& alias
    ) {
      //TODO: Best way to do this?
    };

    virtual void enter_region(Region region, int version) { };
    virtual void  exit_region(Region region, int version) { };

    template<typename F>
    void run_in_region(const std::string& region_label, int version, F&& fun){
      auto last_region = active_region;
      active_region.emplace(get_region(region_label));
      enter_region(active_region.value(), version);
      try {
        fun();
      } catch (...) {
        exit_region(active_region.value(), version);
        if(last_region)active_region.emplace(*last_region);
        else active_region.reset();
        throw;
      }
      exit_region(active_region.value(), version);
      if(last_region)active_region.emplace(*last_region);
      else active_region.reset();
    }

    void register_to(const std::string& region_label, Registration member){
      this->register_member(get_region(region_label), member);
    }
    bool register_if_active(Registration member){
      if(!active_region) return false;
      this->register_member(active_region.value(), member);
      return true;
    }
    void register_to_active(Registration member){
      this->register_member(active_region.value(), member);
    }

    void deregister_from(const std::string& region_label, Registration member){
      this->deregister_member(get_region(region_label), member);
    }
    bool deregister_if_active(Registration member){
      if(!active_region) return false;
      this->deregister_member(active_region.value(), member);
      return true;
    }
    void deregister_from_active(Registration member){
      assert(active_region);
      this->deregister_member(active_region.value(), member);
    }
    
    const FilterFunc& default_filter() const noexcept { return m_default_filter; }

    Config &config() noexcept { return m_config; }
    const Config &config() const noexcept { return m_config; }

#ifdef KR_ENABLE_TRACING
    Util::detail::TraceStack  &trace() { return m_trace; };
#endif

    //Pointer not guaranteed to remain valid, use immediately & discard.
    char* get_scratch_buffer(size_t minimum_size);

    int pid() { return m_pid; }

    Backend& backend() { return m_backend; }

  protected:
    int m_pid;

  private:
    //Hold onto a buffer per context for de/serializing non-contiguous or non-host views.
    std::vector<char> m_scratch_buffer;


    Config m_config;

    FilterFunc m_default_filter;

#ifdef KR_ENABLE_TRACING
    Util::detail::TraceStack  m_trace;
#endif
  protected:
    std::unordered_map<std::string, Members> regions;
    //Get or construct and get a region
    Region get_region(const std::string& label);
    
    std::optional<Region> active_region;

    std::optional<FilterFunc> active_filter;

    Backend m_backend = Impl::make_backend(*this);
    
    //Adds member to the region & informs backend if backend hasn't already
    //registered this member through another region
    virtual void   register_member(Region region, Registration& member);
    virtual void deregister_member(Region region, Registration& member);
  };

  std::unique_ptr< ContextBase > make_context( const std::string &config, int pid );
  std::unique_ptr< ContextBase > make_context( Config cfg, int pid );
#ifdef KR_ENABLE_MPI_CONTEXT
  std::unique_ptr< ContextBase > make_context( MPI_Comm comm, const std::string &cfg );
  std::unique_ptr< ContextBase > make_context( MPI_Comm comm, Config cfg );
#endif
}

#include "resilience/registration/Registration.hpp"
#endif  // INC_RESILIENCE_CONTEXT_HPP
