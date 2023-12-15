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

#ifndef INC_KOKKOS_RESILIENCE_REGISTRATION_VTPROXY_HPP
#define INC_KOKKOS_RESILIENCE_REGISTRATION_VTPROXY_HPP

#include <memory>

#include <vt/vt.h>

#include "resilience/registration/Registration.hpp"
#include "resilience/context/vt/VTContext.hpp"
#include "resilience/util/VTUtil.hpp"

namespace KokkosResilience {
  template<typename T, typename... Traits>
  struct create_registration<T, std::tuple<Traits...>, typename Util::VT::is_proxy<T, void*>::type>{
    std::shared_ptr<Detail::RegistrationBase> reg;

    create_registration(ContextBase& context, T& proxy, std::string label = ""){
      using namespace Context::VT;
        
      label = proxy_label(proxy);

      auto vtCtx = dynamic_cast<VTContext*>(&context);
      if(vtCtx){
        //VTContext handles checkpointing the actual proxy, just register a small metadata member.
        auto& proxy_holder = vtCtx->get_holder(proxy);
        reg = std::make_shared<Detail::MagistrateRegistration<decltype(proxy_holder), BasicCheckpointTrait, Traits...>>
          (proxy_holder, label);

        //If deregistering, vtCtx needs help going from registration to ProxyID
        vtCtx->add_reg_mapping(reg->hash(), proxy);
      } else {
        //Register the full proxy, making sure to include CheckpointTrait
        reg = std::make_shared<Detail::MagistrateRegistration<T, vt::vrt::CheckpointTrait, Traits...>>(proxy, label);
      }
    }

    auto get(){return reg;}
  };
}

#endif
