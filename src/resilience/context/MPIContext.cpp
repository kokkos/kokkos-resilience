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
#include "MPIContext.hpp"
#ifdef KR_ENABLE_VELOC_BACKEND
#include <backend/veloc/VelocBackend.hpp>
#endif
#include <unordered_map>
#include <functional>

namespace KokkosResilience {
std::unique_ptr< ContextBase >
make_context( MPI_Comm comm, const std::string &config )
{
  auto cfg = Config{ config };

  using fun_type = std::function< std::unique_ptr< ContextBase >() >;
  static std::unordered_map< std::string, fun_type > backends = {
#ifdef KR_ENABLE_VELOC_BACKEND
      { "veloc", [&](){ return std::make_unique< MPIContext< VeloCMemoryBackend > >( comm, cfg ); } },
      { "veloc-noop", [&](){ return std::make_unique< MPIContext< VeloCRegisterOnlyBackend > >( comm, cfg ); } }
#endif
  };

  auto pos = backends.find( cfg["backend"].as< std::string >() );
  if ( pos == backends.end() )
    return std::unique_ptr< ContextBase >{};

  return pos->second();
}
}
