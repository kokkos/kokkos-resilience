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

#include "../AutomaticCheckpoint.hpp"

#ifdef KR_ENABLE_TRACING
   #include "../util/Trace.hpp"
#endif

#define VELOC_SAFE_CALL( call ) KokkosResilience::veloc_internal_safe_call( call, #call, __FILE__, __LINE__ )

namespace KokkosResilience
{
  namespace
  {
    void veloc_internal_error_throw( bool success, const char *name, const char *file, int line = 0 )
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

    inline void veloc_internal_safe_call( bool success, const char *name, const char *file, int line = 0 )
    {
      if ( !success )
        veloc_internal_error_throw( success, name, file, line );
    }
  }

  VeloCMemoryBackend::VeloCMemoryBackend(ContextBase &ctx, MPI_Comm mpi_comm)
      : m_context(&ctx), m_last_id(0) {
    const auto &vconf = m_context->config()["backends"]["veloc"]["config"].as< std::string >();
    veloc_client = veloc::get_client(mpi_comm, vconf);
  }

  VeloCMemoryBackend::~VeloCMemoryBackend()
  {
    veloc_client->checkpoint_wait();
  }

  void VeloCMemoryBackend::checkpoint( const std::string &label, int version,
                                       std::set< KokkosResilience::Registration > const &_members )
  {
    bool status = true;

    VELOC_SAFE_CALL( veloc_client->checkpoint_wait() );

    VELOC_SAFE_CALL( veloc_client->checkpoint_begin( label, version ) );

    //checkpoint only unique IDs already registered.
    std::set<int> ids;
    for ( auto member : _members ){
      if(m_registry.find(member) == m_registry.end()){
        //Throw error?
        fprintf(stderr, "KokkosResilience WARNING: Skipping attempt to checkpoint an unregistered member: \"%s\"\n", member->name.c_str());
        continue;
      }
      ids.insert(static_cast<int>(member.hash()));
    }

    VELOC_SAFE_CALL( veloc_client->checkpoint_mem(VELOC_CKPT_SOME, ids) );

    VELOC_SAFE_CALL( veloc_client->checkpoint_end( status ));

    m_latest_version[label] = version;
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
    const std::set< KokkosResilience::Registration > &_members )
  {
    VELOC_SAFE_CALL( veloc_client->restart_begin( label, version ));

    bool status = true;

    //checkpoint only unique IDs already registered.
    std::set<int> ids;
    for ( auto member : _members ){
      ids.insert(static_cast<int>(member.hash()));
fprintf(stderr, "Attempting to recover member %s with id %d\n", member->name.c_str(), static_cast<int>(member.hash()));
    }

    VELOC_SAFE_CALL( veloc_client->recover_mem(VELOC_RECOVER_SOME, ids) );

    VELOC_SAFE_CALL( veloc_client->restart_end( status ) );
  }

  void
  VeloCMemoryBackend::reset()
  {
    for ( auto &&member : m_registry )
    {
      veloc_client->mem_unprotect( static_cast<int>(member.hash()) );
    }

    m_registry.clear();

    m_latest_version.clear();
    m_alias_map.clear();
  }

  void
  VeloCMemoryBackend::register_member(KokkosResilience::Registration &member)
  {
    auto entry = m_registry.insert(member);

    if(!entry.second){
      //Did not insert, already in the set. So don't double mem_protect unless var location has changed
      if((*entry.first)->is_same_reference(member)){
          return;
      }

      //Replace in set.
      m_registry.erase(entry.first);
      m_registry.insert(member);
    }
fprintf(stderr, "%sing member %s with id %d (%lu members in set)\n", entry.second ? "Register" : "Re-register", member->name.c_str(), static_cast<int>(member.hash()), m_registry.size());
    VELOC_SAFE_CALL( veloc_client->mem_protect( static_cast<int>(member.hash()), member->serializer(), member->deserializer() ) );
  }

  void
  VeloCMemoryBackend::register_alias( const std::string &original, const std::string &alias )
  {
    m_alias_map[alias] = original;
  }

  std::string
  VeloCMemoryBackend::get_canonical_label( const std::string &_label ) const noexcept
  {
    // Possible the view has an alias. If so, make sure that is registered instead
    auto pos = m_alias_map.find( _label );
    if ( m_alias_map.end() != pos )
    {
      return pos->second;
    } else {
      return _label;
    }
  }

  void
  VeloCRegisterOnlyBackend::checkpoint( const std::string &label, int version, const std::set<KokkosResilience::Registration> &members )
  {
    // No-op, don't do anything
  }

  void
  VeloCRegisterOnlyBackend::restart(const std::string &label, int version, const std::set<KokkosResilience::Registration> &members)
  {
    // No-op, don't do anything
  }

  VeloCFileBackend::VeloCFileBackend(MPIContext<VeloCFileBackend> &,
                                     MPI_Comm mpi_comm,
                                     const std::string &veloc_config)
      : m_context(&ctx) {
    veloc_client = veloc::get_client( mpi_comm, veloc_config);
  }

  VeloCFileBackend::~VeloCFileBackend()
  {
    veloc_client->checkpoint_wait();
  }

  void
  VeloCFileBackend::checkpoint( const std::string &label, int version,
                                const std::set< KokkosResilience::Registration > &members )
  {
    // Wait for previous checkpoint to finish
    VELOC_SAFE_CALL( veloc_client->checkpoint_wait());

    // Start new checkpoint
    VELOC_SAFE_CALL( veloc_client->checkpoint_begin( label, version ));

    bool status = true;
    try
    {
      std::string fname = veloc_client->route_file("");

      std::ofstream vfile( fname, std::ios::binary );

#ifdef KR_ENABLE_TRACING
      auto write_trace = Util::begin_trace< Util::TimingTrace< std::string > >( *m_context, "write" );
#endif
      for ( auto &&member : members )
      {
          member->serializer()(vfile);
      }
#ifdef KR_ENABLE_TRACING
      write_trace.end();
#endif
    }
    catch ( ... )
    {
      status = false;
    }

    VELOC_SAFE_CALL( veloc_client->checkpoint_end( status ));
  }

  bool
  VeloCFileBackend::restart_available( const std::string &label, int version )
  {
    int latest = veloc_client->restart_test( label, 0 );

    // res is < 0 if no versions available, else it is the latest version
    return version <= latest;
  }

  int VeloCFileBackend::latest_version( const std::string &label ) const noexcept
  {
    return veloc_client->restart_test( label, 0 );
  }

  void VeloCFileBackend::restart( const std::string &label, int version,
                                  const std::set< KokkosResilience::Registration > &members )
  {
    VELOC_SAFE_CALL( veloc_client->restart_begin( label, version ));

    bool status = true;
    try
    {
      std::string fname = veloc_client->route_file("");
      printf( "restore file name: %s\n", fname.c_str() );

      std::ifstream vfile( fname, std::ios::binary );

#ifdef KR_ENABLE_TRACING
      auto read_trace = Util::begin_trace< Util::TimingTrace< std::string > >( *m_context, "read" );
#endif
      for ( auto &&member : members )
      {
          member->deserializer()(vfile);
      }
#ifdef KR_ENABLE_TRACING
      read_trace.end();
#endif
    }
    catch ( ... )
    {
      status = false;
    }

    VELOC_SAFE_CALL( veloc_client->restart_end( status ));
  }
}
