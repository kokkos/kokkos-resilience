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

#ifndef INC_RESILIENCE_REGISTRATION_CUSTOM_HPP
#define INC_RESILIENCE_REGISTRATION_CUSTOM_HPP

#include "Registration.hpp"

namespace KokkosResilience::Impl::RegistrationImpl {
  class Custom : public Base {
  public:
    Custom() = delete;
    Custom(serializer_t&& ser, deserializer_t&& deser, const std::string name) :
        Base(name),
        m_serializer(ser),
        m_deserializer(deser) {};

    const serializer_t serializer() const override{
        return m_serializer;
    }

    const deserializer_t deserializer() const override{
        return m_deserializer;
    }

    const bool is_same_reference(const Registration& other_reg) const override{
      auto other = dynamic_cast<Custom*>(other_reg.get());
      
      if(!other){
        fprintf(stderr,
          "KokkosResilience: Warning, member name %s is shared by more than 1"
          " registration type\n", name.c_str()
        );
        return false;
      }

      return (&m_serializer == &(other->m_serializer)) && 
           (&m_deserializer == &(other->m_deserializer));
    }

  private:
    const serializer_t m_serializer;
    const deserializer_t m_deserializer;
  };
}

#endif //INC_RESILIENCE_REGISTRATION_CUSTOM_HPP
