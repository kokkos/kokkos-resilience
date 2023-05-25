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

#include "resilience/AutomaticCheckpoint.hpp"

#include "resilience/util/Trace.hpp"

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
    
    inline bool veloc_internal_safe_call( bool success, const char *name, const char *file, int line = 0 )
    {
      if ( !success )
        veloc_internal_error_throw( success, name, file, line );
      return success;
    }
  }

  VeloCMemoryBackend::VeloCMemoryBackend(ContextBase* ctx)
      : AutomaticBackendBase(ctx) {
    const auto &vconf = m_context->config()["backends"]["veloc"]["config"].as< std::string >();
    veloc_client = veloc::get_client(ctx->m_pid, vconf);
  }

  VeloCMemoryBackend::~VeloCMemoryBackend()
  {
    veloc_client->checkpoint_wait();
  }
  
  bool VeloCMemoryBackend::checkpoint( const std::string &label, int version,
                                       std::unordered_set< KokkosResilience::Registration > const &_members )
  {
    bool success;
    
    //Don't handle failure here, might be worth trying to continue
    VELOC_SAFE_CALL( veloc_client->checkpoint_wait() );
   
    success = VELOC_SAFE_CALL( veloc_client->checkpoint_begin( label, version ) );

    if(success) {
      std::set<int> ids;
      for(auto member : _members) ids.insert(static_cast<int>(member.hash()));

      success = VELOC_SAFE_CALL( veloc_client->checkpoint_mem(VELOC_CKPT_SOME, ids) );
    }
  
    success = VELOC_SAFE_CALL( veloc_client->checkpoint_end( success ));

    if(success) m_latest_version[label] = version;
    return success;
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
  
  bool
  VeloCMemoryBackend::restart( const std::string &label, int version,
    const std::unordered_set< KokkosResilience::Registration > &_members )
  {
    bool success;
    success = VELOC_SAFE_CALL( veloc_client->restart_begin( label, version ));
    
    if(success){
      std::set<int> ids;
      for(auto member : _members) ids.insert(static_cast<int>(member.hash()));

      success = VELOC_SAFE_CALL( veloc_client->recover_mem(VELOC_RECOVER_SOME, ids) );
    }
    
    success = VELOC_SAFE_CALL( veloc_client->restart_end( success ) );

    return success;
  }

  void
  VeloCMemoryBackend::reset()
  {
    const auto &vconf = m_context->config()["backends"]["veloc"]["config"].as< std::string >();
    veloc_client = veloc::get_client(m_context->m_pid, vconf);

    m_latest_version.clear();
  }
  
  void
  VeloCMemoryBackend::register_member(KokkosResilience::Registration &member)
  {
    veloc_client->mem_protect(
        static_cast<int>(member.hash()), 
        member->serializer(), 
        member->deserializer() 
    );
  }

  VeloCFileBackend::VeloCFileBackend(ContextBase* ctx)
      : AutomaticBackendBase(ctx) {
    const auto &vconf = m_context->config()["backends"]["veloc"]["config"].as< std::string >();
    veloc_client = veloc::get_client( m_context->m_pid, vconf);
  }

  VeloCFileBackend::~VeloCFileBackend()
  {
    veloc_client->checkpoint_wait();
  }
  
  bool
  VeloCFileBackend::checkpoint( const std::string &label, int version,
                                const std::unordered_set< KokkosResilience::Registration > &members )
  {
    bool success;

    // Wait for previous checkpoint to finish
    VELOC_SAFE_CALL( veloc_client->checkpoint_wait());
    
    // Start new checkpoint
    success = VELOC_SAFE_CALL( veloc_client->checkpoint_begin( label, version ));
    
    try
    {
      std::string fname = veloc_client->route_file("");
      
      std::ofstream vfile( fname, std::ios::binary );

      auto write_trace = Util::begin_trace<Util::TimingTrace>( *m_context, "write" );
      for ( auto &&member : members ) {
        success &= member->serializer()(vfile);
        if(!success) break;
      }
      write_trace.end();
    }
    catch ( ... )
    {
      success = false;
    }
    
    success = VELOC_SAFE_CALL( veloc_client->checkpoint_end( success ));
    return success;
  }
  
  bool
  VeloCFileBackend::restart_available( const std::string &label, int version )
  {
    int latest = veloc_client->restart_test( label, version+1 );
    
    // res is < 0 if no versions available, else it is the latest version
    return version == latest;
  }
  
  int VeloCFileBackend::latest_version( const std::string &label ) const noexcept
  {
    return veloc_client->restart_test( label, 0 );
  }
  
  bool VeloCFileBackend::restart( const std::string &label, int version,
                                  const std::unordered_set< KokkosResilience::Registration > &members )
  {
    bool success;
    success = VELOC_SAFE_CALL( veloc_client->restart_begin( label, version ));
   
    if(success) {
      try {
        std::string fname = veloc_client->route_file("");
        
        std::ifstream vfile( fname, std::ios::binary );

        auto read_trace = Util::begin_trace<Util::TimingTrace>( *m_context, "read" );
        for ( auto &&member : members ){
          success = member->deserializer()(vfile);
          if(!success) break;
        }
        read_trace.end();
      } catch ( ... ) {
        success = false;
      }
    }
    
    success = VELOC_SAFE_CALL( veloc_client->restart_end( success ));
    return success;
  }
}
