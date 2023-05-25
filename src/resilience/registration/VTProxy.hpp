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
#include <tuple>
#include "resilience/registration/Registration.hpp"
#include "resilience/context/VTContext.hpp"
#include <vt/vt.h>
#include <checkpoint/checkpoint.h>

namespace KokkosResilience {
namespace Detail {
  template<typename T, typename... Traits>
  struct VTRegistration : public RegistrationBase {
    using VTProxyStatus = VTTemplates::VTProxyStatus;

    VTRegistration(VTContext* ctx, T proxy, const std::string& label) : 
        RegistrationBase(label),
        m_proxy(VTTemplates::get_proxy(proxy)),
        status( ctx->proxy_status(proxy) ) { };

    //We just serialize the version + dependency info of this proxy,
    const serializer_t serializer() const override {
      return [&](std::ostream& stream){
        checkpoint::serializeToStream<Traits...>(status, stream);
        return stream.good();
      };
    }

    const deserializer_t deserializer() const override {
      return [&](std::istream& stream){
        checkpoint::deserializeInPlaceFromStream<Traits...>(stream, &status);
        return stream.good();
      };
    }

    const bool is_same_reference(const Registration& other_reg) const override {
      auto other = dynamic_cast<VTRegistration<T, Traits...>*>(other_reg.get());
      
      if(!other){
        fprintf(stderr, "KokkosResilience: Warning, member name %s is shared by more than 1 registration type\n", name.c_str());
        return false;
      }
      
      return m_proxy == other->m_proxy;
    }

  private:
    const VTProxyType m_proxy;
    VTProxyStatus& status;
  };
}
}

namespace KokkosResilience {
  template<typename T, typename... Traits>
  struct create_registration<T, std::tuple<Traits...>, typename Detail::VTTemplates::is_proxy<T, void*>::type>{
    std::shared_ptr<Detail::RegistrationBase> reg;

    create_registration(ContextBase& context, T proxy, std::string label = ""){
      label = Detail::VTTemplates::get_label(proxy);

      auto vtCtx = dynamic_cast<VTContext*>(&context);
      if(vtCtx){
        reg = std::make_shared<Detail::VTRegistration<T, Traits...>>(vtCtx, proxy, label);
      } else {
        reg = std::make_shared<Detail::MagistrateRegistration<T, vt::vrt::CheckpointTrait, Traits...>>(proxy, label);
      }
    }

    auto get(){return reg;}
  };
}

#endif
