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
#include "FenixBackend.hpp"

#include <sstream>
#include <fstream>
#include <cassert>

#include <fenix.h>

#include <resilience/AutomaticCheckpoint.hpp>

#ifdef KR_ENABLE_TRACING
#include <Resilience/util/Trace.hpp>
#endif

#include <fenix.h>

#define FENIX_SAFE_CALL( call ) KokkosResilience::fenix_internal_safe_call( call, #call, __FILE__, __LINE__ )

namespace KokkosResilience
{
  namespace
  {
    void fenix_internal_error_throw( int e, const char *name, const char *file, int line = 0 )
    {
      std::ostringstream out;
      out << name << " error: Fenix operation failed";
      if ( file )
      {
        out << " " << file << ":" << line;
      }

      // TODO: implement exception class
      //Kokkos::Impl::throw_runtime_exception( out.str() );

      std::cerr << out.str() << std::endl;
    }

    inline void fenix_internal_safe_call( int e, const char *name, const char *file, int line = 0 )
    {
      if ( FENIX_SUCCESS != e )
        fenix_internal_error_throw( e, name, file, line );
    }
  }

  FenixMemoryBackend::FenixMemoryBackend(ContextBase &ctx, MPI_Comm mpi_comm) : m_context(&ctx), m_mpi_comm(mpi_comm)
  {
    int fenix_data_group_id = 1000;
    
    int mpi_comm_size;
    MPI_Comm_size(m_mpi_comm, &mpi_comm_size);

    // TODO: expose policy as user configuration option
    //       isolate Fenix calls inside a fenix client class a la veloc_client_t?
    int fenix_policy_value[3] = { 1, std::max(1, mpi_comm_size / 2), 0 };
    int flag;

    std::cout << "Creating data group " << fenix_data_group_id << '\n';
    FENIX_SAFE_CALL( Fenix_Data_group_create( fenix_data_group_id, m_mpi_comm, 0, 0, FENIX_DATA_POLICY_IN_MEMORY_RAID,
                                              reinterpret_cast<void *>(fenix_policy_value), &flag ) );
  }

  FenixMemoryBackend::~FenixMemoryBackend()
  {
    int fenix_data_group_id = 1000;

    std::cout << "Deleting data group" << fenix_data_group_id << '\n';
    FENIX_SAFE_CALL( Fenix_Data_group_delete( fenix_data_group_id ) );
  }

  void
  FenixMemoryBackend::reset()
  {
    int fenix_data_group_id = 1000;

    for ( auto &&p : m_registry )
    {
      std::cout << "Unprotecting memory id " << p.second.m_id << '\n';
      FENIX_SAFE_CALL( Fenix_Data_member_delete( fenix_data_group_id, p.second.m_id ) );
    }

    std::cout << "Deleting data group" << fenix_data_group_id << '\n';
    FENIX_SAFE_CALL( Fenix_Data_group_delete( fenix_data_group_id ) );

    m_registry.clear();

    m_latest_version.clear();
    m_alias_map.clear();

    int mpi_comm_size;
    MPI_Comm_size(m_mpi_comm, &mpi_comm_size);

    int fenix_policy_value[3] = { 1, std::max(1, mpi_comm_size / 2), 0 };
    int flag;

    std::cout << "Creating data group" << fenix_data_group_id << '\n';
    FENIX_SAFE_CALL( Fenix_Data_group_create( fenix_data_group_id, m_mpi_comm, 0, 0, FENIX_DATA_POLICY_IN_MEMORY_RAID,
                                              reinterpret_cast<void *>(fenix_policy_value), &flag ) );
  }

  std::unordered_set< int >
  FenixMemoryBackend::hash_set( const std::unordered_set< Registration > &members )
  {
    std::unordered_set< int > ids;
    std::transform(
      members.begin(),
      members.end(),
      std::inserter( ids, ids.begin() ),
      [this]( const Registration &member ) -> int
      {
        auto iter = m_alias_map.find( member-> name );
        if ( iter == m_alias_map.end() )
          return static_cast<int>( member->hash() );
        else
          return iter->second;
      }
    );
    return ids;
  }

  void
  FenixMemoryBackend::register_hashes( const std::unordered_set< Registration > &members )
  {
    // Clear protected bits
    for ( auto &&entry : m_registry )
    {
      entry.second.m_protect = false;
    }

    // determine which members should be protected, and store their serialization routines
    for ( auto &&member : members )
    {
      if ( m_alias_map.count( member->name ) != 0 )
        continue;

      auto id = static_cast<int>( member->hash() );

      auto iter = m_registry.find( id );
      if ( iter == m_registry.end() )
      {
        iter = m_registry.emplace( std::piecewise_construct,
                                   std::forward_as_tuple( id ),
                                   std::forward_as_tuple( id, member->serializer(), member->deserializer() ) ).first;
      }

      iter->second.m_protect = true;
    }
  }

  void FenixMemoryBackend::checkpoint( const std::string &label, int version,
                                       const std::unordered_set< Registration > &members )
  {
    int fenix_data_group_id = 1000;

    // register protected members, deregister everyting else
    for ( auto &&p : m_registry )
    {
      if ( p.second.m_protect )
      {
        if ( !p.second.m_registered )
        {
          std::cout << "Protecting member id " << p.second.m_id << '\n';

          // save serialized object to a persistent buffer, which will later be used during actual checkpointing

          std::ostringstream stream;
          p.second.m_serializer(stream);

          std::string temp_buffer = stream.str(); // can we avoid this copy?
          p.second.m_size = temp_buffer.length();

          p.second.m_buffer.resize(p.second.m_size);
          std::memcpy(p.second.m_buffer.data(), temp_buffer.data(), p.second.m_size); // and this copy?

          FENIX_SAFE_CALL( Fenix_Data_member_create( fenix_data_group_id, p.second.m_id, p.second.m_buffer.data(),
                                                     p.second.m_size, MPI_CHAR ) );

          p.second.m_registered = true;
        }
      }
      else
      {
        if ( p.second.m_registered )
        {
          std::cout << "Unprotecting member id " << p.second.m_id << '\n';

          FENIX_SAFE_CALL( Fenix_Data_member_delete( fenix_data_group_id, p.second.m_id ) );

          p.second.m_registered = false;
        }
      }
    }

    const auto ids = hash_set( members );

    for ( auto &&id : ids )
    {
      auto iter = m_registry.find( id );
      if ( iter != m_registry.end() ) 
      {
        std::cout << "Storing member id " << iter->second.m_id << "\n";

        FENIX_SAFE_CALL( Fenix_Data_member_store( fenix_data_group_id, iter->second.m_id, FENIX_DATA_SUBSET_FULL ) );

        // can we safely empty the persistent buffer at this point?
      }
    }

    FENIX_SAFE_CALL( Fenix_Data_commit_barrier( fenix_data_group_id, &version ) );

    m_latest_version[label] = version;
  }

  void
  FenixMemoryBackend::register_alias( const std::string &original, const std::string &alias )
  {
    // TODO double check santization is applied consistently
    m_alias_map[sanitized_label(alias)] = static_cast< int >( label_hash( sanitized_label( original ) ) );
  }

  void
  FenixMemoryBackend::restart( const std::string &label, int version, std::unordered_set< Registration > &members )
  {
    int fenix_data_group_id = 1000;

    const auto ids = hash_set( members );

    for ( auto &&id : ids )
    {
      auto iter = m_registry.find( id );
      if ( iter != m_registry.end() ) 
      {
        std::cout << "Restoring member id " << iter->second.m_id << "\n";

        std::string temp_buffer;
        temp_buffer.resize(iter->second.m_size);

        FENIX_SAFE_CALL( Fenix_Data_member_restore( fenix_data_group_id, iter->second.m_id, temp_buffer.data(),
                                                    iter->second.m_size, version, NULL ) );

        std::istringstream stream( temp_buffer );
        iter->second.m_deserializer( stream );
      }
    }
  }

  bool
  FenixMemoryBackend::restart_available( const std::string &label, int version )
  {
    // res is < 0 if no versions available, else it is the latest version
    return version == latest_version( label );
  }

  int
  FenixMemoryBackend::latest_version( const std::string &label ) const noexcept
  {
    int fenix_data_group_id = 1000;

    auto iter = m_latest_version.find( label );
    if ( iter == m_latest_version.end() )
    {
      int test;
      FENIX_SAFE_CALL( Fenix_Data_group_get_snapshot_at_position( fenix_data_group_id, 0, &test ) );
      m_latest_version[label] = test;
      return test;
    }
    else
    {
      return iter->second;
    }
  }
}
