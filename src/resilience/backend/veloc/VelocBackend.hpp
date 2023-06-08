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
#ifndef INC_RESILIENCE_VELOC_VELOCBACKEND_HPP
#define INC_RESILIENCE_VELOC_VELOCBACKEND_HPP

#include <string>
#include <set>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <mpi.h>
#include "veloc.hpp"

#include <Kokkos_Core.hpp>
#include "resilience/view_hooks/ViewHolder.hpp"
#include "resilience/registration/Registration.hpp"
#include "resilience/context/ContextBase.hpp"
#include "resilience/context/MPIContext.hpp"

#include "resilience/backend/AutomaticBase.hpp"

namespace KokkosResilience
{
  class VeloCMemoryBackend : public AutomaticBackendBase
  {
  public:

    VeloCMemoryBackend(ContextBase& ctx);
    ~VeloCMemoryBackend();

    VeloCMemoryBackend( const VeloCMemoryBackend & ) = delete;
    VeloCMemoryBackend( VeloCMemoryBackend && ) noexcept = default;

    VeloCMemoryBackend &operator=( const VeloCMemoryBackend & ) = delete;
    VeloCMemoryBackend &operator=( VeloCMemoryBackend && ) = default;

    virtual
    bool checkpoint( const std::string &label, int version,
                     const std::unordered_set< Registration > &members,
                     bool as_global) override;


    int latest_version (const std::string &label, int max, bool as_global) const noexcept override;

    virtual
    bool restart( const std::string &label, int version,
                  const std::unordered_set< KokkosResilience::Registration > &members,
                  bool as_global) override;

    void clear_checkpoints();

    void register_member(KokkosResilience::Registration& member) override;
    void unregister_member(const Registration &member) override;

    void reset() override;
    void register_alias( const std::string &original, const std::string &alias );

  private:

    std::string get_canonical_label( const std::string &_label ) const noexcept;

    veloc::client_t *veloc_client;

    mutable std::unordered_map< std::string, int > m_latest_version;

    int m_last_id = 0;
  };

  class VeloCRegisterOnlyBackend : public VeloCMemoryBackend
  {
   public:

    using VeloCMemoryBackend::VeloCMemoryBackend;
    ~VeloCRegisterOnlyBackend() = default;

    VeloCRegisterOnlyBackend( const VeloCRegisterOnlyBackend & ) = delete;
    VeloCRegisterOnlyBackend( VeloCRegisterOnlyBackend && ) noexcept = default;

    VeloCRegisterOnlyBackend &operator=( const VeloCRegisterOnlyBackend & ) = delete;
    VeloCRegisterOnlyBackend &operator=( VeloCRegisterOnlyBackend && ) = default;

    bool checkpoint( const std::string &label, int version,
                     const std::unordered_set< KokkosResilience::Registration > &members, bool as_global) override {
      return true;
    }

    bool restart( const std::string &label, int version,
                  const std::unordered_set< KokkosResilience::Registration > &members, bool as_global) override {
      return true;
    };
  };

  class VeloCFileBackend : AutomaticBackendBase
  {
  public:
    VeloCFileBackend( ContextBase& ctx);
    ~VeloCFileBackend();

    bool checkpoint( const std::string &label, int version,
                     const std::unordered_set< KokkosResilience::Registration > &views,
                     bool as_global);

    int latest_version (const std::string &label, int max, bool as_global) const noexcept;

    bool restart( const std::string &label, int version,
                  const std::unordered_set< KokkosResilience::Registration > &views,
                  bool as_global);

    void register_member( KokkosResilience::Registration & ) {} // Do nothing

  private:
      veloc::client_t* veloc_client;
  };
}

#endif  // INC_RESILIENCE_VELOC_VELOCBACKEND_HPP
