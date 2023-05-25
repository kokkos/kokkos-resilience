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

#include "VTContext.hpp"

namespace KokkosResilience {
std::unique_ptr< ContextBase >
make_context( vt::ctx::Context* vtTheCtx, const std::string &config )
{
  return std::make_unique<VTContext>(config);
}


void VTContext::checkpoint_proxies(){
  using namespace Detail::VTTemplates;

  //Update checkpointed version beforehand, so dependencies
  //also checkpointing now see this version as the required.
  for(auto modified_proxy : modified_proxies){
    auto& status = proxy_registry[modified_proxy];
    status.checkpointed_version++;
  }

  
  //For any remote dependencies we find, query their version
  static int query_version = 0;
  query_version++;

  vt::runInEpochCollective([&]{
    for(auto modified_proxy : modified_proxies){
      auto& status = proxy_registry[modified_proxy];

      for(auto dependency : status.dependencies){
        auto& dep_status = proxy_registry[dependency.first];
        if(!dep_status.is_local() && dep_status.queried_version != query_version) {
          dep_status(m_proxy, REPORT_VERSION);
          dep_status.queried_version = query_version;
        }
      }
    }
  });

  
  //Now actually do the checkpoints to disk.
  for(auto modified_proxy : modified_proxies){
    auto& status = proxy_registry[modified_proxy];

    for(auto dependency : status.dependencies) 
      dependency.second = proxy_registry[dependency.first].checkpointed_version;

    //Non-elements just have their status written
    //in the regular checkpoint steps.
    if(!status.is_element) continue;

    checkpoint_elm(modified_proxy, status.checkpointed_version);
  }

  modified_proxies.clear();
}


}
