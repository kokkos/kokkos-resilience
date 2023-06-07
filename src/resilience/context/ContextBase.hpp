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
#ifndef INC_RESILIENCE_CONTEXTBASE_HPP
#define INC_RESILIENCE_CONTEXTBASE_HPP

#include <Kokkos_Macros.hpp>
#if defined(KOKKOS_ENABLE_HPX)
#include <hpx/config.hpp>
#endif

#include <string>
#include <utility>
#include <memory>
#include <functional>
#include <chrono>
#include <Kokkos_Core.hpp>
#include <unordered_map>
#include <set>

#include "resilience/Config.hpp"
#include "resilience/CheckpointFilter.hpp"
#include "resilience/registration/Registration.hpp"
#include "resilience/view_hooks/ViewHolder.hpp"
#include "resilience/util/Trace.hpp"

#ifdef KR_ENABLE_MAGISTRATE
#include "../registration/Magistrate.hpp"
#endif

namespace KokkosResilience
{
  class ContextBase
  {
  public:
    explicit ContextBase( Config cfg );

    virtual ~ContextBase() {};

    template<typename... Traits, typename RegionFunc, typename FilterFunc, typename... T>
    void run(const std::string& label, int iteration, RegionFunc&& fun, FilterFunc&& filter,
             Detail::RegInfo<T>&... explicit_members);

    template<typename... Traits, typename RegionFunc, typename... T>
    void run(const std::string& label, int iteration, RegionFunc&& fun,
             Detail::RegInfo<T>&... explicit_members) {
      run<Traits...>(label, iteration, std::forward<RegionFunc>(fun), default_filter(), explicit_members...);
    }

    virtual void register_member(KokkosResilience::Registration member) = 0;
    virtual bool restart_available( const std::string &label, int version ) = 0;
    virtual void restart( const std::string &label, int version,
                          const std::set< KokkosResilience::Registration > &members ) = 0;
    virtual void checkpoint( const std::string &label, int version,
                             const std::set< KokkosResilience::Registration > &members ) = 0;

    virtual int latest_version( const std::string &label ) const noexcept = 0;
    virtual void register_alias( const std::string &original, const std::string &alias ) = 0; //TODO: Likely broken with magistrate support changes, double check.

    virtual void reset() = 0;

    virtual void register_members(const std::set< KokkosResilience::Registration > &members) {
      for(auto& member : members) register_member(member);
    };

    //Registers to the active region, requires an active region.
    template<typename... Traits, typename T>
    void register_to_active(T& member, const std::string& label = ""){
      active_region.insert(impl_register<Traits...>(member, label));
    }

    //Registers only if in an active region.
    template<typename... Traits, typename T>
    bool register_if_active(T& member, const std::string& label){
      if(active_region.iter() == regions.end()) return false;
      register_to_active<Traits...>(member, label);
      return true;
    }

    template<typename... Traits, typename T>
    void register_globally(T& member, const std::string& label){
      global_members.insert(impl_register<Traits...>(member, label));
    }

    template<typename... Traits, typename T>
    void register_to(const std::string& region_label, T& member, const std::string& member_label){
      regions[region_label].insert(impl_register<Traits...>(member, member_label));
    }

    const std::function< bool( int ) > &default_filter() const noexcept { return m_default_filter; }

    Config &config() noexcept { return m_config; }
    const Config &config() const noexcept { return m_config; }

    Util::detail::TraceStack  &trace() { return m_trace; };

    //Pointer not guaranteed to remain valid, use immediately & discard.
    char* get_buffer(size_t minimum_size);

    template<typename... Traits>
    void register_to_active(const ViewHolder& view){
      Registration registration = create_registration<ViewHolder, std::tuple<Traits...>>(*this, view);
      register_member(registration); //Virtual function to whatever inheriting class
      active_region.insert(registration);
    }

  protected:
    using RegionsMap = std::unordered_map<std::string, std::set<Registration>>;
    class Region
    {
    public:

      using map_iterator = RegionsMap::iterator;

      Region(map_iterator iter) : m_map_iterator(iter) {};
      const std::string& label() const {
        return m_map_iterator->first;
      }
      const std::set<Registration> &members() const {
        return m_map_iterator->second;
      }
      std::set<Registration>& members(){
        return m_map_iterator->second;
      }
      void insert(const Registration& member){
        members().insert(member);
      }

      auto iter() { return m_map_iterator; }

    private:

      map_iterator m_map_iterator;
    };

    //Create Registration and register to implementation
    //Traits only does anything for Magistrate serialization
    template<typename... Traits, typename T>
    Registration impl_register(T& member, const std::string& label){
      Registration registration = create_registration<T, std::tuple<Traits...>>(*this, member, label);
      register_member(registration); //Virtual function to whatever inheriting class
      return registration;
    }

    template<typename... Traits, typename T>
    void register_to_active(Detail::RegInfo<T>& info){
      register_to_active<Traits...>(info.member, info.label);
    }

  private:
    //Detect views being copied in, register them and any explicitly-listed members.
    template<typename... Traits, typename RegionFunc, typename... T>
    void detect_and_register(RegionFunc&& fun, Detail::RegInfo<T>... explicit_members);

    //Hold onto a buffer per context for de/serializing non-contiguous or non-host views.
    std::vector<char> buffer;

    Config m_config;

    std::function< bool( int ) > m_default_filter;

    Util::detail::TraceStack m_trace;

    RegionsMap regions;
    Region active_region = regions.end();

    //Performance helper
    Region last_region = regions.end();

    std::set<Registration> global_members;

  public:
    static ContextBase* active_context;
  };
}

#endif  // INC_RESILIENCE_CONTEXTBASE_HPP
