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
    void veloc_internal_error_throw( const char *name, const char *file, int line = 0 )
    {
      std::ostringstream out;
      out << name << " error: VELOC operation failed";
      if ( file )
      {
        out << " " << file << ":" << line;
      }

      // TODO: implement exception class
      //Kokkos::Impl::throw_runtime_exception( out.str() );
      // In the meantime, we'll print at least
      out << "\n";
      std::cerr << out.str();
    }

    inline bool veloc_internal_safe_call( bool success, const char *name, const char *file, int line = 0 )
    {
      if ( ! success )
        veloc_internal_error_throw( name, file, line );
      return success;
    }
  }

  VeloCMemoryBackend::VeloCMemoryBackend(ContextBase &ctx, MPI_Comm mpi_comm)
      : m_context(&ctx), m_mpi_comm(mpi_comm) {
    const auto &vconf = m_context->config()["backends"]["veloc"]["config"].as< std::string >();
    veloc_client = veloc::get_client(m_mpi_comm, vconf);
  }

  VeloCMemoryBackend::~VeloCMemoryBackend()
  {
    veloc_client->checkpoint_wait();
  }

  void VeloCMemoryBackend::checkpoint( const std::string &label, int version,
                                       std::unordered_set<Registration> &members )
  {
    VELOC_SAFE_CALL(veloc_client->checkpoint_wait());

    bool success;
    success = VELOC_SAFE_CALL(veloc_client->checkpoint_begin(label, version));
    
    if(success){
      std::set<int> ids = protect_members(members);
      success = VELOC_SAFE_CALL(
        veloc_client->checkpoint_mem(VELOC_CKPT_SOME, ids)
      );
      unprotect_members(ids);
    }

    success = VELOC_SAFE_CALL(veloc_client->checkpoint_end(success));
    if(success) m_latest_version[label] = version;
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
    bool success;
    success = VELOC_SAFE_CALL(veloc_client->restart_begin(label, version));

    if(success){
      std::set<int> ids = protect_members(members);
      success = VELOC_SAFE_CALL(
        veloc_client->recover_mem(VELOC_RECOVER_SOME, ids)
      );
      unprotect_members(ids);
    }

    VELOC_SAFE_CALL( veloc_client->restart_end( success ) );
  }

  void
  VeloCMemoryBackend::reset()
  {
    const auto &vconf = m_context->config()["backends"]["veloc"]["config"].as< std::string >();
    veloc_client = veloc::get_client(m_mpi_comm, vconf);

    m_latest_version.clear();
    m_alias_map.clear();
  }

  int
  VeloCMemoryBackend::protect_member( Registration member )
  {
    auto alias = m_alias_map.find(member->name);
    if(alias != m_alias_map.end()){
      return protect_member(alias->second);
    }
    
    int id = member.hash();
    bool inserted = veloc_client->mem_protect(
      id, member->serializer(), member->deserializer()
    );
    if(!inserted) fprintf(stderr,
      "WARNING KokkosResilience:VeloC memory region %d already existed. "
      "Metadata overwritten by member %s.\n", id, member->name.c_str()
    );

    return id;
  }

  std::set<int>
  VeloCMemoryBackend::protect_members(std::unordered_set<Registration>& members)
  {
    std::set<int> ids;
    for ( auto && member : members ) {
      ids.insert(protect_member(member));
    }
    return ids;
  }

  void
  VeloCMemoryBackend::unprotect_members(const std::set<int>& ids)
  {
    for(auto& id : ids){
      veloc_client->mem_unprotect(id);
    }
  }

  void
  VeloCMemoryBackend::register_alias( Registration& member, const std::string &alias )
  {
    m_alias_map.try_emplace(alias, member);
  }

  VeloCFileBackend::VeloCFileBackend(ContextBase& context, MPI_Comm mpi_comm) 
    : m_context(&context), m_mpi_comm(mpi_comm) {
    const auto &vconf = m_context->config()["backends"]["veloc"]["config"].as< std::string >();
    veloc_client = veloc::get_client(m_mpi_comm, vconf);
  }

  VeloCFileBackend::~VeloCFileBackend()
  {
    veloc_client->checkpoint_wait();
  }

  void
  VeloCFileBackend::checkpoint( const std::string &label, int version,
                                std::unordered_set<Registration> &members )
  {
    // Wait for previous checkpoint to finish
    VELOC_SAFE_CALL( veloc_client->checkpoint_wait() );

    // Start new checkpoint
    VELOC_SAFE_CALL( veloc_client->checkpoint_begin( label, version ) );

    bool success = true;
    try {
      std::string   fname = veloc_client->route_file(label);
      std::ofstream vfile( fname, std::ios::binary );

#ifdef KR_ENABLE_TRACING
      auto write_trace = Util::begin_trace< Util::TimingTrace< std::string > >(
        *m_context, "write"
      );
#endif
      for ( auto& member : members ) {
        success = member->serialize(vfile);
        if(!success) break;
      }
#ifdef KR_ENABLE_TRACING
      write_trace.end();
#endif
    } catch ( const std::exception& e){
      success = false;
      std::cerr << std::string("VelocFileBackend::checkpoint error: ") + e.what();
    } catch ( ... ) {
      success = false;
      std::cerr << "VelocFileBackend::checkpoint error: (unknown exception type)";
    }

    VELOC_SAFE_CALL( veloc_client->checkpoint_end(success) );
  }

  bool
  VeloCFileBackend::restart_available( const std::string &label, int version )
  {
    // res is < 0 if no versions available, else it is the latest version
    return version <= latest_version(label);
  }

  int VeloCFileBackend::latest_version( const std::string &label ) const noexcept
  {
    return veloc_client->restart_test(label, 0);
  }

  void VeloCFileBackend::restart( const std::string &label, int version,
                                  std::unordered_set<Registration> &members )
  {
    VELOC_SAFE_CALL( veloc_client->restart_begin( label, version ));

    bool success = true;
    try {
      std::string   fname = veloc_client->route_file(label);
      std::ifstream vfile( fname, std::ios::binary );

#ifdef KR_ENABLE_TRACING
      auto read_trace = Util::begin_trace< Util::TimingTrace< std::string > >(
        *m_context, "read"
      );
#endif
      for ( auto& member : members ) {
        success = member->deserialize(vfile);
        if(!success) break;
      }
#ifdef KR_ENABLE_TRACING
      read_trace.end();
#endif
    } catch ( const std::exception& e ){
      success = false;
      std::cerr << std::string("VelocFileBackend::restart error: ") + e.what();
    } catch ( ... ) {
      success = false;
      std::cerr << "VelocFileBackend::checkpoint error: (unknown exception type)";
    }

    VELOC_SAFE_CALL( veloc_client->restart_end(success) );
  }
}
