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
#ifndef INC_RESILIENCE_BACKEND_VELOC_HPP
#define INC_RESILIENCE_BACKEND_VELOC_HPP

#include <string>
#include <set>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <mpi.h>
#include "veloc.hpp"

#include <Kokkos_Core.hpp>

#include "resilience/registration/Registration.hpp"
#include "resilience/backend/Backend.hpp"
#include "resilience/config/Config.hpp"

namespace KokkosResilience::Impl::BackendImpl
{
  class VeloC : public Base
  {
   public:
    VeloC(ContextBase& ctx);
    ~VeloC();

    bool checkpoint(
      const std::string &label, int version, const Members& members
    ) override;
    
    int latest_version(const std::string &label, int max) const override;
 
    bool restart(
      const std::string &label, int version, const Members& members
    ) override;

    void reset() override;

   protected:
    // Register the members and return the set of their IDs to checkpoint
    std::set<int> register_members(const Members& members);
    void deregister_members(const std::set<int>& ids);

    veloc::client_t *veloc_client;
    
    mutable std::unordered_map< std::string, int > m_latest_version;

    Config m_conf;
    const std::string& veloc_config_file = m_conf["config"].as<std::string>();
    const bool checkpoint_to_file = m_conf.get("file", false);
  };
}

#endif  // INC_RESILIENCE_BACKEND_VELOC_HPP
