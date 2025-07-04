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
#ifndef INC_RESILIENCE_MEMPROTECT_HPP
#define INC_RESILIENCE_MEMPROTECT_HPP

#include <cstdint>
#include <vector>

namespace KokkosResilience {

namespace Detail
  {
    struct MemProtectKey
    {
      explicit MemProtectKey( void *maddr )
          : addr( reinterpret_cast< std::uintptr_t >( maddr ) )
      {}

      std::uintptr_t addr;

      friend bool operator==( const MemProtectKey &_lhs, const MemProtectKey &_rhs )
      {
        return _lhs.addr == _rhs.addr;
      }

      friend bool operator!=( const MemProtectKey &_lhs, const MemProtectKey &_rhs )
      {
        return !( _lhs == _rhs );
      }

      friend bool operator<( const MemProtectKey &_lhs, const MemProtectKey &_rhs )
      {
        return _lhs.addr < _rhs.addr;
      }
    };

    struct MemProtectBlock
    {
      explicit MemProtectBlock( int mid )
          : id( mid )
      {}

      int id;
      std::vector< unsigned char > buff;
      void *ptr = nullptr;
      std::size_t size = 0;
      std::size_t element_size = 0;
      bool protect = false;
      bool registered = false;
    };
  }

}

namespace std
{
  template<>
  struct hash< KokkosResilience::Detail::MemProtectKey >
  {
    std::size_t operator()( const KokkosResilience::Detail::MemProtectKey &_mem ) const noexcept
    {
      return std::hash< std::uintptr_t >{}( _mem.addr );
    }
  };
}

#endif
