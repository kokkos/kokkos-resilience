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
      : m_context(&ctx), m_last_id(0) {
    const auto &vconf = m_context->config()["backends"]["veloc"]["config"].as< std::string >();
    VELOC_SAFE_CALL( VELOC_Init( mpi_comm, vconf.c_str() ) );
  }

  VeloCMemoryBackend::~VeloCMemoryBackend()
  {
    VELOC_Checkpoint_wait();
    VELOC_Finalize( false );
  }

  void VeloCMemoryBackend::checkpoint( const std::string &label, int version,
                                       const std::vector< KokkosResilience::ViewHolder > &_views )
  {
    bool status = true;

    // Check if we need to copy any views to backing store
    for ( auto &&view : _views )
    {
      std::string label = get_canonical_label( view->label() );

      if ( !view->span_is_contiguous() || !view->is_host_space() )
      {
        auto pos = m_registry.find( label );
        if ( pos != m_registry.end())
        {
          view->deep_copy_to_buffer( pos->second.buff.data() );
          assert( pos->second.buff.size() == view->data_type_size() * view->span() );
        }
      }
    }

    VELOC_SAFE_CALL( VELOC_Checkpoint_wait() );

    VELOC_SAFE_CALL( VELOC_Checkpoint_begin( label.c_str(), version ) );

    VELOC_SAFE_CALL( VELOC_Checkpoint_mem() );

    VELOC_SAFE_CALL( VELOC_Checkpoint_end( status ));

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
    auto lab = get_canonical_label( label );
    auto latest_iter = m_latest_version.find( lab );
    if ( latest_iter == m_latest_version.end() )
    {
      auto test = VELOC_Restart_test(lab.c_str(), 0);
      m_latest_version[lab] = test;
      return test;
    } else {
     return latest_iter->second;
    }
  }

  void
  VeloCMemoryBackend::restart( const std::string &label, int version,
    const std::vector< KokkosResilience::ViewHolder > &_views )
  {
    auto lab = get_canonical_label( label );
    VELOC_SAFE_CALL( VELOC_Restart_begin( lab.c_str(), version ));

    bool status = true;

    VELOC_SAFE_CALL( VELOC_Recover_mem() );

    VELOC_SAFE_CALL( VELOC_Restart_end( status ) );

    // Check if we need to copy any views from the backing store back to the view
    for ( auto &&view : _views )
    {
      auto vl = get_canonical_label( view->label() );
      if ( !view->span_is_contiguous() || !view->is_host_space() )
      {
        auto pos = m_registry.find( vl );
        if ( pos != m_registry.end() )
        {
          assert( pos->second.buff.size() == view->data_type_size() * view->span() );
          view->deep_copy_from_buffer( pos->second.buff.data() );
        }
      }
    }
  }

  void
  VeloCMemoryBackend::reset()
  {
    for ( auto &&vr : m_registry )
    {
      VELOC_Mem_unprotect( vr.second.id );
    }

    m_registry.clear();

    m_latest_version.clear();
    m_alias_map.clear();
  }

  void
  VeloCMemoryBackend::register_hashes( const std::vector< KokkosResilience::ViewHolder > &views,
                                       const std::vector< Detail::CrefImpl > &crefs  )
  {
    // Clear protected bits
    for ( auto &&p : m_registry )
    {
      p.second.protect = false;
    }

    for ( auto &&view : views )
    {
      if ( !view->data() )  // uninitialized view
        continue;

      std::string label = get_canonical_label( view->label() );
      auto iter = m_registry.find( label );

      // Attempt to find the view in our registry
      if ( iter == m_registry.end() )
      {
        // Calculate id using hash of view label
        int id = ++m_last_id; // Prefix since we will consider id 0 to be no-id
        iter = m_registry.emplace( std::piecewise_construct,
                                        std::forward_as_tuple( label ),
                                        std::forward_as_tuple( id ) ).first;
        iter->second.element_size = view->data_type_size();
        iter->second.size = view->span();

        if ( !view->is_host_space() || !view->span_is_contiguous() )
        {
          // Can't reference memory directly, allocate memory for a watch buffer
          iter->second.buff.assign( iter->second.size * iter->second.element_size, 0x00 );
          iter->second.ptr = iter->second.buff.data();
        } else {
          iter->second.ptr = view->data();
        }
      }

      // iter now pointing to our entry
      iter->second.protect = true;
    }

    // Register crefs
    for ( auto &&cref : crefs )
    {
      if ( !cref.ptr )  // uninitialized view
        continue;
      // If we haven't already register, register with VeloC
      auto iter = m_registry.find( cref.name );
      if ( iter == m_registry.end())
      {
        int id = ++m_last_id; // Prefix since we will consider id 0 to be no-id
        iter = m_registry.emplace( std::piecewise_construct,
            std::forward_as_tuple( cref.name ),
            std::forward_as_tuple( id ) ).first;

        iter->second.ptr = cref.ptr;
        iter->second.size = cref.num;
        iter->second.element_size = cref.sz;
      }

      iter->second.protect = true;
    }

    // Register everything protected, unregister anything unprotected
    for ( auto &&p : m_registry )
    {
      if ( p.second.protect )
      {
        if ( !p.second.registered )
        {
          std::cout << "Protecting memory id " << p.second.id << " with label " << p.first << '\n';
          VELOC_SAFE_CALL( VELOC_Mem_protect( p.second.id, p.second.ptr, p.second.size, p.second.element_size ) );
          p.second.registered = true;
        }
      } else { //deregister
        if ( p.second.registered )
        {
          std::cout << "Unprotecting memory id " << p.second.id << " with label " << p.first << '\n';
          VELOC_Mem_unprotect( p.second.id );
          p.second.registered = false;
        }
      }
    }
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
  VeloCRegisterOnlyBackend::checkpoint( const std::string &label, int version, const std::vector<KokkosResilience::ViewHolder> &views )
  {
    // No-op, don't do anything
  }

  void
  VeloCRegisterOnlyBackend::restart(const std::string &label, int version, const std::vector<KokkosResilience::ViewHolder> &views)
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
                                const std::vector< KokkosResilience::ViewHolder > &views )
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
      for ( auto &&v : views )
      {
        char        *bytes = static_cast< char * >( v->data());
        std::size_t len    = v->span() * v->data_type_size();

        vfile.write( bytes, len );
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
                                  const std::vector< KokkosResilience::ViewHolder > &views )
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
      for ( auto &&v : views )
      {
        char        *bytes = static_cast< char * >( v->data());
        std::size_t len    = v->span() * v->data_type_size();

        vfile.read( bytes, len );
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
