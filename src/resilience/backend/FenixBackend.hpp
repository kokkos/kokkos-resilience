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
#ifndef INC_RESILIENCE_FENIX_FENIXBACKEND_HPP
#define INC_RESILIENCE_FENIX_FENIXBACKEND_HPP

#include <string>
#include <vector>
#include <memory>
#include <Kokkos_Core.hpp>
#include <unordered_map>
#include <unordered_set>
#include <mpi.h>
#include "resilience/registration/Registration.hpp"

namespace KokkosResilience
{
  class ContextBase;

  class FenixClient;

  class FenixMemoryBackend
  {
  public:

    FenixMemoryBackend( ContextBase &ctx, MPI_Comm mpi_comm );
    ~FenixMemoryBackend();

    FenixMemoryBackend( const FenixMemoryBackend & ) = delete;
    FenixMemoryBackend( FenixMemoryBackend && ) noexcept = default;

    FenixMemoryBackend &operator=( const FenixMemoryBackend & ) = delete;
    FenixMemoryBackend &operator=( FenixMemoryBackend && ) = default;

    void register_hashes( const std::unordered_set< Registration > &members );

    void checkpoint( const std::string &label, int version, const std::unordered_set< Registration > &members );

    int latest_version (const std::string &label ) const noexcept;

    bool restart_available( const std::string &label, int version );

    void restart( const std::string &label, int version, std::unordered_set< Registration > &members );

    void register_alias( const std::string &original, const std::string &alias );

    void clear_checkpoints();

    void reset();

  private:
    std::unordered_set< int > hash_set( const std::unordered_set< Registration > &members );

    ContextBase *m_context;
    FenixClient *m_client;
    mutable std::unordered_map< std::string, int > m_latest_version;
    std::unordered_map< std::string, int > m_alias_map;
  };
}

#endif  // INC_RESILIENCE_FENIX_FENIXBACKEND_HPP
