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
#ifndef INC_KOKKOS_RESILIENCE_RESILIENTREF_HPP
#define INC_KOKKOS_RESILIENCE_RESILIENTREF_HPP

#include <utility>
#include <vector>
#include <memory>

namespace KokkosResilience
{
  struct in_place_t
  {
    explicit in_place_t() = default;
  };

  template< typename T >
  class Ref
  {
   public:

    using reference = T &;
    using pointer = T *;

    explicit Ref( reference _obj )
      : m_object( &_obj ) {

    }

    Ref( Ref &&_other ) noexcept = default;
    Ref &operator=( Ref &&_other ) noexcept = default;

    Ref( const Ref &_other )
      : m_object( _other.m_object ) {
      // Now explicitly copy the object to a temporary
      auto copy = *m_object;
    }

    Ref &operator=( const Ref &_other ) {
      m_object = _other.m_object;
      auto copy = *m_object;
    }

    reference get() const noexcept { return *m_object; }
    reference operator*() const noexcept { return *m_object; }
    pointer operator->() const noexcept { return m_object; }

   private:

    T *m_object;
  };

  template< typename T, typename... Args >
  Ref< T > make_ref( Args &&... _args )
  {
    return Ref< T >( in_place_t{}, std::forward< Args >( _args )... );
  }
}

#endif  // KOKKOS_RESILIENCE_RESILIENTREF_HPP
