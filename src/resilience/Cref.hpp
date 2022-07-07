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
#ifndef INC_RESILIENCE_CREF_HPP
#define INC_RESILIENCE_CREF_HPP

#include <vector>

namespace KokkosResilience
{
  namespace Detail
  {
    struct CrefImpl
    {
      CrefImpl( void *p, std::size_t s, std::size_t n, const char *_name )
          : ptr( p ), sz( s ), num( n ), name( _name )
      {}

      void *ptr;
      std::size_t sz;
      std::size_t num;
      const char *name;
    };

    struct Cref : public CrefImpl
    {
      using CrefImpl::CrefImpl;

      Cref( const Cref &_other )
          : CrefImpl( _other.ptr, _other.sz, _other.num, _other.name )
      {
        if ( check_ref_list )
          check_ref_list->emplace_back( ptr, sz, num, name );
      }

      static std::vector< CrefImpl > *check_ref_list;
    };
  }

  template< typename T >
  auto check_ref( T &_t, const char *_str )
  {
    return Detail::Cref{ reinterpret_cast< void * >( &_t ), sizeof( T ), 1, _str };
  }
}

#endif  // INC_RESILIENCE_CREF_HPP
