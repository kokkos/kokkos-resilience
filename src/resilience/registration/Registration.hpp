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

#include <string>
#include <functional>
#include <cmath>
#include <climits>
#include <memory>

namespace KokkosResilience
{ 

  //Takes a stream as input, returns success flag
  using   serializer_t = std::function<bool (std::ostream &)>;
  using deserializer_t = std::function<bool (std::istream &)>;

  struct Registration;

  namespace Detail {
    std::string sanitized_label(std::string label);
    size_t label_hash(const std::string& label);

    struct RegistrationBase {
      explicit RegistrationBase(const std::string member_name) : 
          name(sanitized_label(member_name)) { }

      RegistrationBase() = delete;
      virtual ~RegistrationBase() = default;

      virtual const serializer_t serializer() const = 0;
      virtual const deserializer_t deserializer() const = 0;
      virtual const bool is_same_reference(const Registration&) const = 0;

      const bool operator==(const RegistrationBase& other) const;
      virtual const size_t hash() const;
      
      const std::string name;
    };
    
    
    //Helper for explicitly-listing data that a 
    //checkpoint region should also use.
    template<typename T>
    struct RegInfo {
      RegInfo(T& member, const std::string label) : member(member), label(label) {};
      T& member;
      const std::string label;
    };
  }

  
  //A struct convertible to Registration, use as if function returning Registration.
  //Generally, register as: create_registration(ContextBase* ctx, T& member, const std::string& label);
  //But see various registration headers for any specializations based on member type
  template<typename T, typename Traits = std::tuple<>, typename enable = void*>
  struct create_registration;

  //Make registration using custom (de)serialize functions
  Registration custom_registration(serializer_t&& s_fun, deserializer_t&& d_fun, const std::string label);

  struct Registration : public std::shared_ptr<Detail::RegistrationBase> {
    template<typename RegType>
    Registration(std::shared_ptr<RegType> base) 
      : std::shared_ptr<Detail::RegistrationBase>(std::move(base)) {}

    //Auto-convert create_registration structs into Registrations
    template<typename... T>
    Registration(create_registration<T...> reg) 
      : Registration(reg.get()) {};

    const size_t hash() const;
    const bool operator==(const Registration& other) const;
  };
} //namespace KokkosResilience

namespace std {
  template<>
  struct hash<KokkosResilience::Registration>{
    size_t operator()(const KokkosResilience::Registration& registration) const {
      return registration.hash();
    }
  };
}

#include "./Simple.hpp"
#include "./ViewHolder.hpp"
#include "./Custom.hpp"

#endif //_INC_RESILIENCE_REGISTRATION_HPP
