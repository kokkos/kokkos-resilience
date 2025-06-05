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
#include <fenix.h>
#include <cassert>

#include <resilience/AutomaticCheckpoint.hpp>
#include <resilience/backend/VelocBackend.hpp>

#ifdef KR_ENABLE_TRACING
#include <Resilience/util/Trace.hpp>
#endif

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
    }

    inline void fenix_internal_safe_call( int e, const char *name, const char *file, int line = 0 )
    {
      if ( FENIX_SUCCESS != e )
        fenix_internal_error_throw( e, name, file, line );
    }
  }

  FenixMemoryBackend::FenixMemoryBackend(ContextBase &ctx, MPI_Comm mpi_comm)
      : m_context(&ctx), m_last_id(0), m_last_group(0), m_fenix_data_group_id(1234) {
    MPI_Comm_dup(mpi_comm, &m_mpi_comm);
    std::cout << "Creating data group\n";
    FENIX_SAFE_CALL( Fenix_Data_group_create( m_fenix_data_group_id, m_mpi_comm, 0, 0, m_fenix_policy_name, (void *)(m_fenix_policy_value), &m_fenix_policy_flag ) );
  }

  FenixMemoryBackend::~FenixMemoryBackend()
  {
    std::cout << "Deleting data group\n";
    FENIX_SAFE_CALL( Fenix_Data_group_delete( m_fenix_data_group_id ) );
    MPI_Comm_free(&m_mpi_comm);
  }

  void
  FenixMemoryBackend::reset()
  {
    for ( auto &&vr : m_registry )
    {
      std::cout << "Unprotecting memory id " << vr.second.id << " with label " << vr.first << '\n';
      FENIX_SAFE_CALL( Fenix_Data_member_delete( m_fenix_data_group_id, vr.second.id ) );
    }

    std::cout << "Deleting data group\n";
    FENIX_SAFE_CALL( Fenix_Data_group_delete( m_fenix_data_group_id ) );

    m_registry.clear();

    m_latest_version.clear();
    m_alias_map.clear();

    std::cout << "Creating data group\n";
    FENIX_SAFE_CALL( Fenix_Data_group_create( m_fenix_data_group_id, m_mpi_comm, 0, 0, m_fenix_policy_name, (void *)(m_fenix_policy_value), &m_fenix_policy_flag ) );
  }

  void
  FenixMemoryBackend::register_hashes( const std::vector< KokkosResilience::ViewHolder > &views,
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
          FENIX_SAFE_CALL( Fenix_Data_member_create( m_fenix_data_group_id, p.second.id, p.second.ptr, p.second.size * p.second.element_size, MPI_CHAR ) );
          p.second.registered = true;
        }
      } else { //deregister
        if ( p.second.registered )
        {
          std::cout << "Unprotecting memory id " << p.second.id << " with label " << p.first << '\n';
          FENIX_SAFE_CALL( Fenix_Data_member_delete( m_fenix_data_group_id, p.second.id ) );
          p.second.registered = false;
        }
      }
    }
  }

  void
  FenixMemoryBackend::register_alias( const std::string &original, const std::string &alias )
  {
    m_alias_map[alias] = original;
  }

  void FenixMemoryBackend::checkpoint( const std::string &label, int version,
                                       const std::vector< KokkosResilience::ViewHolder > &_views )
  {
    for ( auto &&view : _views )
    {
      std::string label = get_canonical_label( view->label() );
      auto pos = m_registry.find( label );

      if ( pos != m_registry.end())
      {
        // Check if we need to copy any views to backing store
        if ( !view->span_is_contiguous() || !view->is_host_space() )
        {
          view->deep_copy_to_buffer( pos->second.buff.data() );
          assert( pos->second.buff.size() == view->data_type_size() * view->span() );
        }

        // store data to fenix
        std::cout << "Storing memory id " << pos->second.id << " with label " << pos->first << '\n';
        FENIX_SAFE_CALL( Fenix_Data_member_store(m_fenix_data_group_id, pos->second.id, FENIX_DATA_SUBSET_FULL) );
      }
    }

    FENIX_SAFE_CALL( Fenix_Data_commit_barrier( m_fenix_data_group_id, &version ) );

    m_latest_version[label] = version;
  }

  void
  FenixMemoryBackend::restart( const std::string &label, int version,
    const std::vector< KokkosResilience::ViewHolder > &_views )
  {
    auto lab = get_canonical_label( label );

    for ( auto &&view : _views )
    {
      auto vl = get_canonical_label( view->label() );
      auto pos = m_registry.find( vl );

      if ( pos != m_registry.end() )
      {
        // restore data from fenix
        std::cout << "Restoring memory id " << pos->second.id << " with label " << pos->first << '\n';
        FENIX_SAFE_CALL( Fenix_Data_member_restore(m_fenix_data_group_id, pos->second.id, pos->second.ptr, pos->second.size * pos->second.element_size, version, NULL) );

        // Check if we need to copy any views from the backing store back to the view
        if ( !view->span_is_contiguous() || !view->is_host_space() )
        {
          assert( pos->second.buff.size() == view->data_type_size() * view->span() );
          view->deep_copy_from_buffer( pos->second.buff.data() );
        }
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
    auto lab = get_canonical_label( label );
    auto latest_iter = m_latest_version.find( lab );
    if ( latest_iter == m_latest_version.end() )
    {
      int test;
      FENIX_SAFE_CALL( Fenix_Data_group_get_snapshot_at_position( m_fenix_data_group_id, 0, &test ) );
      m_latest_version[lab] = test;
      return test;
    } else {
     return latest_iter->second;
    }
  }

  std::string
  FenixMemoryBackend::get_canonical_label( const std::string &_label ) const noexcept
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
}
