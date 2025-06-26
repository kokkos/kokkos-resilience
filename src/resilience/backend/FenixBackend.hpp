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
#include "../view_hooks/ViewHolder.hpp"
#include <unordered_map>
#include <unordered_set>
#include <mpi.h>
#include <fenix.h>
#include "../Cref.hpp"

namespace KokkosResilience
{
  class ContextBase;

  namespace Detail
  {
    struct ProtectedMemoryBlock
    {
      explicit ProtectedMemoryBlock( int mid ) : id( mid ) {}

      int id;
      std::vector< unsigned char > buff;
      void *ptr = nullptr;
      std::size_t size = 0;
      std::size_t element_size = 0;
      bool protect = false;
      bool registered = false;
    };
  }

  class FenixMemoryBackend
  {
  public:

    FenixMemoryBackend( ContextBase &ctx, MPI_Comm mpi_comm );
    ~FenixMemoryBackend();

    FenixMemoryBackend( const FenixMemoryBackend & ) = delete;
    FenixMemoryBackend( FenixMemoryBackend && ) noexcept = default;

    FenixMemoryBackend &operator=( const FenixMemoryBackend & ) = delete;
    FenixMemoryBackend &operator=( FenixMemoryBackend && ) = default;

    void checkpoint( const std::string &label, int version,
                     const std::vector< KokkosResilience::ViewHolder > &views );

    bool restart_available( const std::string &label, int version );
    int latest_version (const std::string &label) const noexcept;

    void restart( const std::string &label, int version,
                  const std::vector< KokkosResilience::ViewHolder > &views );

    void clear_checkpoints();

    void register_hashes( const std::vector< KokkosResilience::ViewHolder > &views,
      const std::vector< Detail::CrefImpl > &crefs );

    void reset();
    void register_alias( const std::string &original, const std::string &alias );

  private:

    std::string get_canonical_label( const std::string &_label ) const noexcept;

    std::unordered_map< std::string, Detail::ProtectedMemoryBlock > m_registry;

    MPI_Comm m_mpi_comm;
    ContextBase *m_context;

    mutable std::unordered_map< std::string, int > m_latest_version;
    std::unordered_map< std::string, std::string > m_alias_map;
    int m_last_id;

    const int m_fenix_data_group_id = 1000;
    const int m_fenix_policy_name = FENIX_DATA_POLICY_IN_MEMORY_RAID;
    const int m_fenix_policy_value[3] = {1, 1, 0}; // pairs ranks (0, 1), (2, 3), (4, 5), ...
    int m_fenix_policy_flag;
  };
}

#endif  // INC_RESILIENCE_FENIX_FENIXBACKEND_HPP
