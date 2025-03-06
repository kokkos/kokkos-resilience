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

#ifndef _INC_RESILIENCE_REGISTRATION_HPP
#define _INC_RESILIENCE_REGISTRATION_HPP

#include <string>
#include <functional>
#include <cmath>
#include <climits>
#include <memory>
#include <iostream>

#include "resilience/context/Context.hpp"

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
      explicit RegistrationBase(const std::string member_name);

      RegistrationBase() = delete;
      virtual ~RegistrationBase() = default;

      const bool serialize(std::ostream& out);
      const bool deserialize(std::istream& in);

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
      RegInfo(T& m_member, const std::string m_label);
      T& member;
      const std::string label;
    };

    //Enables overrides for the default SimpleRegistration
    template<typename T, typename enable = void*>
    struct SpecializedRegistration;
  }

  struct Registration {
    Registration(std::shared_ptr<Detail::RegistrationBase> m_base) 
      : base(m_base) {}

    //Typical constructor
    template<typename T>
    Registration(ContextBase& ctx, T& member, const std::string& label);

    //For members we can infer labels for
    template<typename T>
    Registration(ContextBase& ctx, T& member);
    
    template<typename T>
    Registration(ContextBase& ctx, Detail::RegInfo<T>& reg_info); 

    //Create using custom (de)serialize functions
    Registration(serializer_t&& s_fun, deserializer_t&& d_fun, const std::string& label);

    const size_t hash() const;
    const bool operator==(const Registration& other) const;
    Detail::RegistrationBase& operator*() const {
      return *base.get();
    }
    Detail::RegistrationBase* operator->() const {
      return base.get();
    }
    Detail::RegistrationBase* get() const {
      return base.get();
    }

  private:
    std::shared_ptr<Detail::RegistrationBase> base;
  };
} //namespace KokkosResilience


#include "Registration.impl.hpp"
#include "Simple.hpp"
#include "ViewHolder.hpp"
#include "Custom.hpp"

#endif //_INC_RESILIENCE_REGISTRATION_HPP
