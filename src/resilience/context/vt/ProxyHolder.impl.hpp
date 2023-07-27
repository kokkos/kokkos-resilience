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

#ifndef INC_KOKKOS_RESILIENCE_CONTEXT_VT_PROXYHOLDER_IMPL_HPP
#define INC_KOKKOS_RESILIENCE_CONTEXT_VT_PROXYHOLDER_IMPL_HPP

#include "ProxyHolder.hpp"

namespace KokkosResilience::Context::VT {
template<typename ProxyT>
ProxyHolder::ProxyHolder(ProxyT proxy, VTContext& context)
  : ProxyID(proxy), 
    data_reg(build_registration(proxy)),
    ctx(&context)
{
  invoker = [this, proxy](ProxyAction action, std::any arg) {
    return ctx->action_handler(proxy, *this, action, arg);
  };
  ctx->m_backend->register_member(metadata_registration());
  ctx->m_backend->register_member(data_registration());

  if constexpr(is_col<ProxyT>::value and not is_elm<ProxyT>::value){
    using ObjT = typename elm_type<ProxyT>::type;

    using EventT = vt::vrt::collection::listener::ElementEventEnum;
    using IndexT = typename ObjT::IndexType;
    listener_id = vt::theCollection()->registerElementListener<ObjT, IndexT>(
        proxy_bits,
        [this, proxy](EventT event, IndexT index, vt::NodeType elm_home){
          ctx->handle_element_event(proxy[index], event);
          return;
        }
    );
  }
};

template<typename SerT>
void ProxyHolder::serialize(SerT& s){
  [[maybe_unused]] const auto old_proxy_bits = proxy_bits;
  s | proxy_bits | _status;
  assert(old_proxy_bits == proxy_bits);

/*if(!s.hasTraits(BasicCheckpointTrait()) && (s.isPacking() || s.isUnpacking())) 
  fmt::print("{}: {} status {} to version {}. Tracked: {}\n", ctx->m_proxy, *this, s.isPacking() ? "Packed" : "Unpacked", _status.checkpointed_version, tracked());
if(s.hasTraits(BasicCheckpointTrait()) && (s.isPacking() || s.isUnpacking())) 
  fmt::print("{}: {} basic status {} to version {}\n", ctx->m_proxy, *this, s.isPacking() ? "Packed" : "Unpacked", _status.checkpointed_version);*/
}

}

#endif
