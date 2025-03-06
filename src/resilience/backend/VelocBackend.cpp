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
#include "VelocBackend.hpp"

#include <sstream>
#include <fstream>
#include <veloc.h>
#include <cassert>
#include <unordered_set>

#include <resilience/context/MPIContext.hpp>
#include <resilience/AutomaticCheckpoint.hpp>

#ifdef KR_ENABLE_TRACING
   #include <Resilience/util/Trace.hpp>
#endif

#define VELOC_SAFE_CALL( call ) KokkosResilience::veloc_internal_safe_call( call, #call, __FILE__, __LINE__ )

namespace KokkosResilience
{
  namespace
  {
    void veloc_internal_error_throw( int e, const char *name, const char *file, int line = 0 )
    {
      std::ostringstream out;
      out << name << " error: VELOC operation failed";
      if ( file )
      {
        out << " " << file << ":" << line;
      }

      // TODO: implement exception class
      //Kokkos::Impl::throw_runtime_exception( out.str() );
    }

    inline void veloc_internal_safe_call( int e, const char *name, const char *file, int line = 0 )
    {
      if ( VELOC_SUCCESS != e )
        veloc_internal_error_throw( e, name, file, line );
    }
  }

  VeloCMemoryBackend::VeloCMemoryBackend(ContextBase &ctx, MPI_Comm mpi_comm)
      : m_context(&ctx), m_mpi_comm(mpi_comm) {
    const auto &vconf = m_context->config()["backends"]["veloc"]["config"].as< std::string >();
    veloc_client = veloc::get_client(m_mpi_comm, vconf.c_str());
  }

  VeloCMemoryBackend::~VeloCMemoryBackend()
  {
    veloc_client->checkpoint_wait();
  }

  void VeloCMemoryBackend::checkpoint( const std::string &label, int version,
                                       std::unordered_set<Registration> &members )
  {
    VELOC_SAFE_CALL(veloc_client->checkpoint_wait());

    VELOC_SAFE_CALL( veloc_client->checkpoint_begin( label, version ) );

    VELOC_SAFE_CALL( 
        veloc_client->checkpoint_mem(VELOC_CKPT_SOME, hash_set(members))
    );

    bool success = true;
    VELOC_SAFE_CALL( veloc_client->checkpoint_end(success) ); 

    m_latest_version[label] = version;
  }
    
  std::set<int> VeloCMemoryBackend::hash_set(std::unordered_set<Registration> &members){
    std::set<int> ids;
    std::transform(members.begin(), members.end(), std::inserter(ids, ids.begin()), 
        [this](const Registration& member) -> int {
          auto iter = m_alias_map.find(member->name);
          if (iter == m_alias_map.end())
            return static_cast<int>(member->hash());
          else 
            return iter->second;
        });
    return ids;
  }

  bool
  VeloCMemoryBackend::restart_available( const std::string &label, int version )
  {
    // res is < 0 if no versions available, else it is the latest version
    return version == latest_version( label );
  }

  int
  VeloCMemoryBackend::latest_version( const std::string &label ) const noexcept
  {
    auto latest_iter = m_latest_version.find( label );
    if ( latest_iter == m_latest_version.end() )
    {
      auto test = veloc_client->restart_test(label, 0);
      m_latest_version[label] = test;
      return test;
    } else {
     return latest_iter->second;
    }
  }

  void
  VeloCMemoryBackend::restart( const std::string &label, int version,
                               std::unordered_set<Registration> &members )
  {
    VELOC_SAFE_CALL( veloc_client->restart_begin( label, version ));

    VELOC_SAFE_CALL( veloc_client->recover_mem(VELOC_RECOVER_SOME, hash_set(members)) );

    bool status = true;
    VELOC_SAFE_CALL( veloc_client->restart_end( status ) );
  }

  void
  VeloCMemoryBackend::reset()
  {
    const auto &vconf = m_context->config()["backends"]["veloc"]["config"].as< std::string >();
    veloc_client = veloc::get_client(m_mpi_comm, vconf);

    m_latest_version.clear();
    m_alias_map.clear();
  }

  void
  VeloCMemoryBackend::register_hashes( std::unordered_set<Registration> &members )
  {
    for ( auto && member : members ) {
      if(m_alias_map.count(member->name) != 0)
        continue;

      // Not a "safe" call, since we don't care if this was 
      //   or wasn't already registered
      veloc_client->mem_unprotect(
            static_cast<int>(member.hash())
      );
      VELOC_SAFE_CALL(veloc_client->mem_protect(
            static_cast<int>(member.hash()),
            member->serializer(),
            member->deserializer()
      ));
    }
  }

  void
  VeloCMemoryBackend::register_alias( const std::string &original, const std::string &alias )
  {
    m_alias_map[Detail::sanitized_label(alias)] = static_cast<int>(
        Detail::label_hash(Detail::sanitized_label(original))
    );
  }

  void
  VeloCRegisterOnlyBackend::checkpoint( const std::string &label, int version,
      std::unordered_set<Registration> &members )
  {
    // No-op, don't do anything
  }

  void
  VeloCRegisterOnlyBackend::restart(const std::string &label, int version,
      std::unordered_set<Registration> &members)
  {
    // No-op, don't do anything
  }

  VeloCFileBackend::VeloCFileBackend(MPIContext<VeloCFileBackend> &,
                                     MPI_Comm mpi_comm,
                                     const std::string &veloc_config) {
    VELOC_SAFE_CALL( VELOC_Init( mpi_comm, veloc_config.c_str()));
  }

  VeloCFileBackend::~VeloCFileBackend()
  {
    VELOC_Finalize( false );
  }

  void
  VeloCFileBackend::checkpoint( const std::string &label, int version,
                                std::unordered_set<Registration> &members )
  {
    // Wait for previous checkpoint to finish
    VELOC_SAFE_CALL( VELOC_Checkpoint_wait());

    // Start new checkpoint
    VELOC_SAFE_CALL( VELOC_Checkpoint_begin( label.c_str(), version ));

    char veloc_file_name[VELOC_MAX_NAME];

    bool status = true;
    try
    {
      VELOC_SAFE_CALL( VELOC_Route_file( veloc_file_name, veloc_file_name ) );

      std::string   fname( veloc_file_name );
      std::ofstream vfile( fname, std::ios::binary );

#ifdef KR_ENABLE_TRACING
      auto write_trace = Util::begin_trace< Util::TimingTrace< std::string > >( *m_context, "write" );
#endif
      for ( auto &&member : members )
      {
        status = member->serialize(vfile);
        if(!status) break;
      }
#ifdef KR_ENABLE_TRACING
      write_trace.end();
#endif
    }
    catch ( ... )
    {
      status = false;
    }

    VELOC_SAFE_CALL( VELOC_Checkpoint_end( status ));
  }

  bool
  VeloCFileBackend::restart_available( const std::string &label, int version )
  {
    int latest = VELOC_Restart_test( label.c_str(), 0 );

    // res is < 0 if no versions available, else it is the latest version
    return version <= latest;
  }

  int VeloCFileBackend::latest_version( const std::string &label ) const noexcept
  {
    return VELOC_Restart_test( label.c_str(), 0 );
  }

  void VeloCFileBackend::restart( const std::string &label, int version,
                                  std::unordered_set<Registration> &members )
  {
    VELOC_SAFE_CALL( VELOC_Restart_begin( label.c_str(), version ));

    char veloc_file_name[VELOC_MAX_NAME];

    bool status = true;
    try
    {
      VELOC_SAFE_CALL( VELOC_Route_file( veloc_file_name, veloc_file_name ) );
      printf( "restore file name: %s\n", veloc_file_name );

      std::string   fname( veloc_file_name );
      std::ifstream vfile( fname, std::ios::binary );

#ifdef KR_ENABLE_TRACING
      auto read_trace = Util::begin_trace< Util::TimingTrace< std::string > >( *m_context, "read" );
#endif
      for ( auto &&member : members )
      {
        status = member->deserialize(vfile);
        if(!status) break;
      }
#ifdef KR_ENABLE_TRACING
      read_trace.end();
#endif
    }
    catch ( ... )
    {
      status = false;
    }

    VELOC_SAFE_CALL( VELOC_Restart_end( status ));
  }
}
