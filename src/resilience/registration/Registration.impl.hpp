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

#ifndef _INC_RESILIENCE_REGISTRATION_IMPL_HPP
#define _INC_RESILIENCE_REGISTRATION_IMPL_HPP

#include "Registration.hpp"

namespace KokkosResilience {
  namespace Detail {
    template<typename T>
    RegInfo<T>::RegInfo(T& m_member, const std::string m_label) :
      member(m_member), label(m_label) {};


    template<typename T, typename enable>
    struct SpecializedRegistration {
      //Specializations must exist
      static constexpr bool exists = false;
      
      //Specializations must implement this, but may ignore provided label.
      SpecializedRegistration(ContextBase& ctx, T& member, const std::string& label);
      //Specializations may implement this, if label can be inferred.
      SpecializedRegistration(ContextBase& ctx, T& member);

      std::shared_ptr<RegistrationBase> get();
    };

    template<typename T>
    struct SimpleRegistration;
  }

  template<typename T>
  Registration::Registration(
    ContextBase& ctx, T& member, const std::string& label
  ) : base(nullptr)
  {
    if constexpr(Detail::SpecializedRegistration<T>::exists) {
      base = Detail::SpecializedRegistration(ctx, member, label).get();
    } else {
      base = std::make_shared<Detail::SimpleRegistration<T>>(member, label);
    }
  }

  template<typename T>
  Registration::Registration(
    ContextBase& ctx, T& member
  ) : Registration(Detail::SpecializedRegistration(ctx, member).get())
  { }

  template<typename T>
  Registration::Registration(ContextBase& ctx, Detail::RegInfo<T>& reg_info) 
    : Registration(ctx, reg_info.member, reg_info.label)
  { }
}


namespace std {
  template<>
  struct hash<KokkosResilience::Registration>{
    size_t operator()(const KokkosResilience::Registration& registration) const {
      return registration.hash();
    }
  };
}

#endif //_INC_RESILIENCE_REGISTRATION_HPP
