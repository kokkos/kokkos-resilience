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
#include <resilience/util/Trace.hpp>
#endif

#include <fenix.h>

// TODO: implement exception class
//Kokkos::Impl::throw_runtime_exception( out.str() );
#define FENIX_THROW( msg ) std::cerr << msg << std::endl

namespace KokkosResilience
{
  class FenixClient
  {
    using serializer_t   = std::function< bool( std::ostream & ) >;
    using deserializer_t = std::function< bool( std::istream & ) >;

    struct ProtectedMemoryBlock
    {
      explicit ProtectedMemoryBlock( int id, const serializer_t &serializer, const deserializer_t &deserializer )
          : m_id( id ), m_serializer( serializer ), m_deserializer( deserializer ) {}

      int m_id;
      serializer_t m_serializer;
      deserializer_t m_deserializer;
      std::vector<char> m_buffer;
      std::size_t m_size;
      bool m_protect = false;
      bool m_registered = false;
    };
    
  public:
    // TODO Enable user configuration options, especially in regards to data group policy
    FenixClient( MPI_Comm comm ) : m_comm( comm )
    {
      create_data_group();
    }

    ~FenixClient()
    {
      clear_registry();
      delete_data_group();
    }

    void protect_member( int hash, serializer_t serializer, deserializer_t deserializer )
    {
      int member_id = g_current_member_id++;

      auto item = m_registry.find( hash );
      if ( item == m_registry.end() )
      {
        item = m_registry.emplace( std::piecewise_construct,
                                   std::forward_as_tuple( hash ),
                                   std::forward_as_tuple( member_id, serializer, deserializer ) ).first;
      }

      item->second.m_protect = true;
    }

    void unprotect_member( int hash )
    {
      auto item = m_registry.find( hash );
      if ( item != m_registry.end() && item->second.m_registered )
      {
        Fenix_Data_member_delete( m_group_id, item->second.m_id );
        item->second.m_protect = false;
        item->second.m_registered = false;
      }
      m_registry.erase( item );
    }

    void checkpoint( const std::unordered_set< int > &hashes, int &version )
    {
      for ( auto &&hash : hashes )
      {
        auto item = m_registry.find( hash );
        if ( item != m_registry.end() )
        {
          // check if we need to create new Fenix data member
          if (item->second.m_protect && !item->second.m_registered)
          {
            // load serialized object into stream
            std::ostringstream stream;
            item->second.m_serializer(stream);

            // get data out of stream and save size for later recovery
            std::string temp_buffer = stream.str(); // can we avoid this copy?
            item->second.m_size = temp_buffer.length();

            // copy data into persistent buffer associated with member id
            item->second.m_buffer.resize(item->second.m_size);
            std::memcpy(item->second.m_buffer.data(), temp_buffer.data(), item->second.m_size);
            temp_buffer.clear();

            int status = Fenix_Data_member_create( m_group_id, item->second.m_id, item->second.m_buffer.data(),
                                                   item->second.m_size, MPI_CHAR );

            if (status != FENIX_SUCCESS)
            {
              std::ostringstream msg;
              msg << "error: failed to create Fenix data member " << item->second.m_id << " in data group "
                  << m_group_id << " (return code = " << status << ")";
              FENIX_THROW( msg.str() );
            }

            item->second.m_registered = true;
          }

          int status = Fenix_Data_member_store( m_group_id, item->second.m_id, FENIX_DATA_SUBSET_FULL );

          if (status != FENIX_SUCCESS)
          {
            std::ostringstream msg;
            msg << "error: failed to store Fenix data member " << item->second.m_id << " in data group "
                << m_group_id << " (return code = " << status << ")";
            FENIX_THROW( msg.str() );
          }

          // safely clear persistent buffers?
        }
      }

      int status = Fenix_Data_commit_barrier( m_group_id, &version );

      if (status != FENIX_SUCCESS)
      {
        std::ostringstream msg;
        msg << "error: failed to commit data members in data group " << m_group_id << " (return code = " << status
            << ")";
        FENIX_THROW( msg.str() );
      }
    }

    void restart( const std::unordered_set< int > &hashes, int version )
    {
      for ( auto &&hash : hashes )
      {
        auto item = m_registry.find( hash );
        if ( item != m_registry.end() )
        {
          std::string temp_buffer;
          temp_buffer.resize(item->second.m_size);

          int status = Fenix_Data_member_restore( m_group_id, item->second.m_id, temp_buffer.data(),
                                                  item->second.m_size, version, NULL );
          
          if ( status != FENIX_SUCCESS )
          {
            std::ostringstream msg;
            msg << "error: failed to restore data member " << item->second.m_id << " in data group " << m_group_id
                << " (return code = " << status << ")";
            FENIX_THROW( msg.str() );
          }

          std::istringstream stream( temp_buffer );
          item->second.m_deserializer( stream );
        }
      }
    }

    int latest_version( const std::string &label )
    {
      int test;
      int status = Fenix_Data_group_get_snapshot_at_position( m_group_id, 0, &test );

      if ( status != FENIX_SUCCESS )
      {
        std::ostringstream msg;
        msg << "error: failed to determine latest version for data group " << m_group_id << " (return code = "
            << status << ")";
        FENIX_THROW( msg.str() );
      }

      return test;
    }

    void reset()
    {
      clear_registry();
      delete_data_group();
      create_data_group();
    }

  private:
    void create_data_group()
    {
      int comm_size;
      MPI_Comm_size(m_comm, &comm_size);

      int policy_value[3] = {  1, std::max(1, comm_size / 2), 0 };
      int flag;

      m_group_id = g_current_group_id++;
      int status = Fenix_Data_group_create( m_group_id, m_comm, 0, 0, FENIX_DATA_POLICY_IN_MEMORY_RAID,
                                            reinterpret_cast<void *>(policy_value), &flag );
      if ( status != FENIX_SUCCESS ) {
        std::ostringstream msg;
        msg << "error: failed to create Fenix data group " << m_group_id << " (return code = " << status
            << ", flag = " << flag << ")";
        FENIX_THROW( msg.str() );
      }
    }

    void delete_data_group()
    {
      Fenix_Data_group_delete( m_group_id );
    }

    void clear_registry()
    {
      for ( auto &&item : m_registry )
      {
        unprotect_member( item.first );
      }
      m_registry.clear();
    }

    static int g_current_group_id;
    static int g_current_member_id;

    MPI_Comm m_comm;
    int m_group_id;
    std::unordered_map< int, ProtectedMemoryBlock > m_registry;
  };

  int FenixClient::g_current_group_id = 1000;
  int FenixClient::g_current_member_id = 1000;

  FenixMemoryBackend::FenixMemoryBackend(ContextBase &ctx, MPI_Comm mpi_comm)
    : m_context(&ctx), m_client(new FenixClient( mpi_comm ))
  {
  }

  FenixMemoryBackend::~FenixMemoryBackend()
  {
    delete m_client;
  }

  void
  FenixMemoryBackend::reset()
  {
    m_client->reset();
    m_latest_version.clear();
    m_alias_map.clear();
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
        auto iter = m_alias_map.find( member->name );
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
    for ( auto &&member : members )
    {
      if ( m_alias_map.count( member->name ) != 0 )
        continue;

      auto hash = static_cast<int>( member->hash() );
      m_client->unprotect_member( hash );
      m_client->protect_member( hash, member->serializer(), member->deserializer() );
    }
  }

  void FenixMemoryBackend::checkpoint( const std::string &label, int version,
                                       const std::unordered_set< Registration > &members )
  {
    m_client->checkpoint( hash_set( members ), version );
    m_latest_version[label] = version;
  }

  void
  FenixMemoryBackend::register_alias( const std::string &original, const std::string &alias )
  {
    m_alias_map[sanitized_label(alias)] = static_cast< int >( label_hash( sanitized_label( original ) ) );
  }

  void
  FenixMemoryBackend::restart( const std::string &label, int version, std::unordered_set< Registration > &members )
  {
    m_client->restart( hash_set( members ), version );
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
    auto iter = m_latest_version.find( label );
    if ( iter == m_latest_version.end() )
    {
      int test = m_client->latest_version( label );
      m_latest_version[label] = test;
      return test;
    }
    else
    {
      return iter->second;
    }
  }
}
