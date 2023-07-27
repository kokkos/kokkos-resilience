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

#ifndef INC_KOKKOS_RESILIENCE_CONTEXT_VT_PROXYMAP_HPP
#define INC_KOKKOS_RESILIENCE_CONTEXT_VT_PROXYMAP_HPP

#include <optional>
#include "common.hpp"
#include "ProxyHolder.hpp"

namespace KokkosResilience::Context::VT {

class ProxyMap {
public:
  ProxyMap(VTContext& context) : ctx(context) {};
 
  //Get existing, or initialize and get proxy holder for this proxy.
  template <
    typename ProxyT,
    typename enable = typename is_proxy<ProxyT>::type
  >
  ProxyHolder& operator[](ProxyT proxy);

  //Get existing, or attempt to initialize. May return nullptr;
  ProxyHolder* operator[](ProxyID proxy_id){
    auto iter = id_to_holder.find(proxy_id);
    if(iter != id_to_holder.end()) return &(iter->second);
    
    auto group_iter = group_to_member_id.find(proxy_id.proxy_bits);
    if(group_iter == group_to_member_id.end()) return nullptr;
    
    auto& group_holder = id_to_holder[group_iter->second];
    return group_holder.get_holder(proxy_id);
  }

  //If a registration's hash matches a held proxy's, return pointer to it.
  ProxyHolder* operator[](Registration& reg){
    auto iter = hash_to_id.find(reg.hash());
    if(iter == hash_to_id.end()) return nullptr;

    return (*this)[iter->second];
  }

  void add_reg_mapping(size_t reg_hash, ProxyID proxy_id){
    hash_to_id[reg_hash] = proxy_id;
  }

  std::unordered_map<ProxyID, ProxyHolder>& map(){
    return id_to_holder;
  }

private:
  VTContext& ctx;

  std::unordered_map<ProxyID, ProxyHolder> id_to_holder;

  //Find a proxy ID from the hash of its (core) registration.
  std::unordered_map<size_t, ProxyID> hash_to_id;

  //Find a representative member of a group by its proxy bits
  std::unordered_map<uint64_t, ProxyID> group_to_member_id;
};

}
#endif
