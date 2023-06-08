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

#ifndef INC_KOKKOS_RESILIENCE_REGISTRATION_MAGISTRATE_HPP
#define INC_KOKKOS_RESILIENCE_REGISTRATION_MAGISTRATE_HPP

#include "resilience/registration/Registration.hpp"
#include <checkpoint/checkpoint.h>

#ifdef KR_ENABLE_VT
#include "resilience/util/VTUtil.hpp"
#endif

namespace KokkosResilience::Detail {
  //Registration for some type which Magistrate knows how to checkpoint.
  template
  <
    typename MemberType,
    typename... Traits
  >
  struct MagistrateRegistration : public RegistrationBase {
    MagistrateRegistration() = delete;

    MagistrateRegistration(MemberType& member, std::string name)
      : RegistrationBase(name), m_member(member) {}

    const serializer_t serializer() const override{
      return [&, this](std::ostream& stream){
        checkpoint::serializeToStream<
          Traits...
        >(m_member, stream);
        return stream.good();
      };
    }

    const deserializer_t deserializer() const override{
      return [&, this](std::istream& stream){
        checkpoint::deserializeInPlaceFromStream<
          Traits...
        >(stream, &m_member);
        return stream.good();
      };
    }

    const bool is_same_reference(const Registration& other_reg) const override{
      auto other = dynamic_cast<MagistrateRegistration<MemberType, Traits...>*>(other_reg.get());

      if(!other){
        //We wouldn't expect this to happen, and it may indicate a hash collision
        fprintf(stderr, "KokkosResilience: Warning, member name %s is shared by more than 1 registration type\n", name.c_str());
        return false;
      }

      return &m_member == &other->m_member;
    }

  private:
    MemberType& m_member;
  };
}


namespace KokkosResilience {
  template<
    typename T,
    typename... Traits
  >
  struct create_registration<
    T,
    std::tuple<Traits...>,
    std::enable_if_t<
      checkpoint::SerializableTraits<T>::is_traversable
#ifdef KR_ENABLE_VT
      and not Util::VT::is_proxy<T>::value
#endif
    >*
  > {
    using BaseT = Detail::MagistrateRegistration<T, Traits...>;
    std::shared_ptr<BaseT> reg;

    create_registration(ContextBase& ctx, T& member, std::string label)
        : reg(std::make_shared<BaseT>(member, label)) {};

    auto get() && {
      return std::move(reg);
    }
  };
}

#endif  // INC_RESILIENCE_MAGISTRATE_HPP
