#include "VelocBackend.hpp"

#include <sstream>
#include <fstream>
#include <veloc.h>
#include <cassert>
#include <unistd.h>

#include "../MPIContext.hpp"
#include "../AutomaticCheckpoint.hpp"

#ifdef KR_ENABLE_TRACING
   #include "../util/Trace.hpp"
#endif

//#define KR_ENABLE_INCREMENTAL_HASH
//#define KR_ENABLE_INCREMENTAL_SCAN

#ifdef KR_ENABLE_INCREMENTAL_HASH
#include "etl/murmur3.h"
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
  
  VeloCMemoryBackend::VeloCMemoryBackend( ContextBase &ctx, MPI_Comm mpi_comm )
    : m_mpi_comm( mpi_comm ), m_context( &ctx ), m_last_id( 0 )
  {
    const auto &vconf = m_context->config()["backends"]["veloc"]["config"].as< std::string >();
    VELOC_SAFE_CALL( VELOC_Init( mpi_comm, vconf.c_str() ) );
  }
  
  VeloCMemoryBackend::~VeloCMemoryBackend()
  {
    std::cout << "Destroy backend\n";
    VELOC_Checkpoint_wait();
    VELOC_Finalize( false );
  }
  
  void VeloCMemoryBackend::checkpoint( const std::string &label, int version,
                                       const std::vector< std::unique_ptr< Kokkos::ViewHolderBase > > &_views )
  {
    std::cout << "Create checkpoint " << label << std::endl;
    bool status = true;
    
    // Check if we need to copy any views to backing store
    for ( auto &&view : _views )
    {
      std::string label = get_canonical_label( view->label() );

      if ( !view->span_is_contiguous() || !view->is_hostspace() )
      {
        auto pos = m_registry.find( label );
        if ( pos != m_registry.end())
        {
          view->deep_copy_to_buffer( pos->second.buff.data() );
          assert( pos->second.buff.size() == view->data_type_size() * view->span() );
        }
      }
    }

#ifdef KR_ENABLE_INCREMENTAL_SCAN
    for ( auto &&view : _views )
    {
      std::string label = get_canonical_label( view->label() );
      auto pos = m_registry.find( label );
      if ( pos != m_registry.end() && pos->second.incremental )
      {
        uint32_t element_size = pos->second.element_size;
        uint32_t counter = 0;
std::cout << "Checkpointing " << label << " with " << view->span() << " entries and " << view->span()*view->data_type_size() << " bytes" << std::endl;
        for(uint64_t i=0; i<pos->second.size; i++) 
        {
          if(memcmp((uint8_t*)(view->data())+i*(element_size), 
                    pos->second.incr_buff.data()+i*(element_size), 
                    element_size) != 0)
          {
            pos->second.indx_buff[counter] = i;
            memcpy(pos->second.incr_buff.data()+counter*(element_size), (uint8_t*)(view->data())+i*(element_size), element_size);
            counter += 1;
          }
        }
        pos->second.num_changes = counter;
        std::cout << counter << " changes, " << counter*view->data_type_size() << "/" << view->span()*view->data_type_size() << " bytes updated " << std::endl;
        if ( counter > 0 ) 
        {
VELOC_Mem_unprotect(pos->second.id);
VELOC_Mem_unprotect(pos->second.indx_id);
VELOC_Mem_unprotect(pos->second.meta_data_id);
          VELOC_SAFE_CALL( VELOC_Mem_protect( pos->second.id, pos->second.incr_buff.data(), counter, element_size) );
          VELOC_SAFE_CALL( VELOC_Mem_protect( pos->second.indx_id, pos->second.indx_buff.data(), counter, sizeof(uint32_t)) );
          VELOC_SAFE_CALL( VELOC_Mem_protect( pos->second.meta_data_id, &(pos->second.num_changes), 1, sizeof(uint32_t)) );
        }
      }
    }
#endif

#ifdef KR_ENABLE_INCREMENTAL_HASH
    // Handle incremental checkpoints
    size_t page_size = sysconf(_SC_PAGE_SIZE);
    etl::murmur3<uint32_t> murmur3_32_gen;
    for ( auto &&view : _views )
    {
      std::string label = get_canonical_label( view->label() );
//std::cout << "Checkpoint " << label << " ?\n";
      bool updated = false;
      auto pos = m_registry.find( label );
      if ( pos != m_registry.end() && pos->second.incremental )
      {
//std::cout << "Found View " << label << std::endl;
        // Calculate # of pages for the View
        int num_pages = view->span()*view->data_type_size()/page_size;
        if(num_pages*page_size < view->span()*view->data_type_size())
          num_pages+=1;
//std::cout << "Number of pages: " << num_pages << std::endl;
        
        // Counter for the number of dirty pages
        size_t counter = 0;
        size_t num_bytes = 0;
        // Iterate and save all dirty pages
        for ( size_t i=0; i<num_pages; i++)
        {
          // Calculate how many bytes to write. Necessary for the last page
          size_t num_write = std::min(page_size, view->span()*view->data_type_size() - page_size*i);
          // Load data into hash generator
          murmur3_32_gen.add((unsigned char*)(view->data())+(i*page_size), (unsigned char*)(view->data())+((i)*page_size)+num_write);
          uint32_t hash_val = murmur3_32_gen.value();
//std::cout << "Generated hash\n";
          // If a change is detected
          if (hash_val != pos->second.meta_data[i])
          {
            updated = true;
            // Update hash list
            pos->second.meta_data[i] = hash_val;
//std::cout << "Saving page " << i << " in " << counter << " th place" << std::endl;
            // Copy hash+page into buffer
            memcpy(pos->second.incr_buff.data()+counter*(sizeof(uint32_t)+page_size), &i, sizeof(uint32_t));
            memcpy(pos->second.incr_buff.data()+counter*(sizeof(uint32_t)+page_size)+sizeof(uint32_t), 
                    (unsigned char*)(view->data())+i*page_size, num_write);
//std::cout << "Copied data\n";
            // Update # of changed pages
            counter += 1;
            num_bytes += num_write+sizeof(uint32_t);
          }
          murmur3_32_gen.reset();
//std::cout << "Reset hash gen\n";
        }
        pos->second.meta_data[pos->second.meta_data.size()-2] = counter;
        pos->second.meta_data[pos->second.meta_data.size()-1] = num_bytes;
        if ( updated ) 
        {
          VELOC_Mem_unprotect( pos->second.id );
          VELOC_SAFE_CALL( VELOC_Mem_protect( pos->second.id, pos->second.incr_buff.data(), num_bytes, 1) );
          VELOC_SAFE_CALL( VELOC_Mem_protect( pos->second.meta_data_id, pos->second.meta_data.data(), pos->second.meta_data.size(), sizeof(uint32_t)) );
        }
        std::cout << counter << "/" << num_pages << " pages updated " << num_bytes << " total bytes" <<std::endl;
      }
    }
#endif
    
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
    const std::vector< std::unique_ptr< Kokkos::ViewHolderBase > > &_views )
  {
    std::cout << "Restart checkpoint " << label << " version " << version << std::endl;

    size_t page_size = sysconf(_SC_PAGE_SIZE);
    std::vector<size_t> incremental_indices;
    std::vector<size_t> non_incremental_indices;
    std::vector<int> incremental_ids;
    std::vector<int> metadata_ids;
    std::vector<int> non_incremental_ids;
    for(size_t i=0; i<_views.size(); i++) 
    {
      auto pos = m_registry.find(get_canonical_label(_views[i]->label()));
      if (pos != m_registry.end() && pos->second.incremental && pos->second.protect)
      {
        //std::cout << "Found incremental view: " << _views[i]->label() << std::endl;
        incremental_indices.push_back(i);
        incremental_ids.push_back(pos->second.id);
#ifdef KR_ENABLE_INCREMENTAL_SCAN
        incremental_ids.push_back(pos->second.indx_id);
#endif
#if defined(KR_ENABLE_INCREMENTAL_SCAN) || defined(KR_INCREMENTAL_HASH)
        metadata_ids.push_back(pos->second.meta_data_id);
        pos->second.incr_buff.assign(_views[i]->span() * _views[i]->data_type_size(), 0);
        pos->second.indx_buff.assign(_views[i]->span(), 0);
        pos->second.changed_buff.assign(pos->second.changed_buff.size(), true);
#endif
      }
      else if (pos != m_registry.end() && pos->second.protect)
      {
        non_incremental_indices.push_back(i);
        non_incremental_ids.push_back(pos->second.id);
      }
    }

    // Restart regular views
    auto lab = get_canonical_label( label );
    bool status = true;
    
    VELOC_SAFE_CALL( VELOC_Restart_begin( lab.c_str(), version ));
    
    VELOC_SAFE_CALL( VELOC_Recover_selective(VELOC_RECOVER_SOME, non_incremental_ids.data(), non_incremental_ids.size()) );
    
    VELOC_SAFE_CALL( VELOC_Recover_selective(VELOC_RECOVER_SOME, metadata_ids.data(), metadata_ids.size()) );
    
    VELOC_SAFE_CALL( VELOC_Restart_end( status ) );

    //std::cout << "Restored metadata views\n";
  
    // Check if we need to copy any views from the backing store back to the view
    for ( auto &&view : _views )
    {
      auto vl = get_canonical_label( view->label() );
      if ( !view->span_is_contiguous() || !view->is_hostspace() )
      {
        auto pos = m_registry.find( vl );
        if ( pos != m_registry.end() )
        {
          assert( pos->second.buff.size() == view->data_type_size() * view->span() );
          view->deep_copy_from_buffer( pos->second.buff.data() );
        }
      }
    }

#ifdef KR_ENABLE_INCREMENTAL_SCAN
    if(incremental_ids.size() > 0)
    {
      int current_version = version;
      while(current_version >= 0)
      {
        // Protect the incremental views
        for(size_t i=0; i<incremental_indices.size(); i++)
        {
          auto&& view = _views[incremental_indices[i]];
          auto pos = m_registry.find(get_canonical_label(view->label()));
          if(pos != m_registry.end())
          {
            VELOC_SAFE_CALL( VELOC_Mem_protect( pos->second.id, pos->second.incr_buff.data(), view->span()*view->data_type_size(), 1 ) );
            VELOC_SAFE_CALL( VELOC_Mem_protect( pos->second.indx_id, pos->second.indx_buff.data(), view->span(), view->data_type_size() ) );
          }
        }
        VELOC_SAFE_CALL( VELOC_Restart_begin( lab.c_str(), current_version ) );
        VELOC_SAFE_CALL( VELOC_Recover_selective(VELOC_RECOVER_SOME, incremental_ids.data(), incremental_ids.size()) );
        status = true;
        VELOC_SAFE_CALL( VELOC_Restart_end( status ) );
        VELOC_SAFE_CALL( VELOC_Restart_begin( lab.c_str(), current_version ) );
        VELOC_SAFE_CALL( VELOC_Recover_selective(VELOC_RECOVER_SOME, metadata_ids.data(), metadata_ids.size()) );
        status = true;
        VELOC_SAFE_CALL( VELOC_Restart_end( status ) );
        for(size_t i=0; i<incremental_indices.size(); i++)
        {
          auto&& view = _views[incremental_indices[i]];
          auto vl = get_canonical_label( view->label() );
          auto pos = m_registry.find( vl );
          if ( pos != m_registry.end() )
          {
            std::cout << pos->second.num_changes << " Changes, " << std::endl;
            for( int j=0; j<pos->second.num_changes; j++ )
            {
              if(pos->second.changed_buff[pos->second.indx_buff[j]])
              {
//                *((unsigned char*)(view->data())) = pos->second.incr_buff[pos->second.indx_buff[j]];
                memcpy((uint8_t*)(view->data())+pos->second.indx_buff[j]*view->data_type_size(),
                      pos->second.incr_buff.data()+view->data_type_size()*j,
                      view->data_type_size());
                pos->second.changed_buff[pos->second.indx_buff[j]] = false;
              }
            }
          }
        }
        current_version -= 1;
      }
    }
#endif

#ifdef KR_ENABLE_INCREMENTAL_HASH
    if(incremental_ids.size() > 0) 
    {
      int current_version = version;
      while(current_version >= 0)
      {
        // Protect the incremental views
        for(size_t i=0; i<incremental_indices.size(); i++)
        {
          auto&& view = _views[incremental_indices[i]];
          auto pos = m_registry.find(get_canonical_label(view->label()));
          if(pos != m_registry.end())
          {
            int num_pages = view->span()*view->data_type_size()/page_size;
            if(num_pages*page_size < view->span()*view->data_type_size())
              num_pages+=1;
            VELOC_SAFE_CALL( VELOC_Mem_protect( pos->second.id, pos->second.incr_buff.data(), view->span()*view->data_type_size()+num_pages*sizeof(uint32_t), 1 ) );
//            VELOC_SAFE_CALL( VELOC_Mem_protect( pos->second.id, pos->second.incr_buff.data(), pos->second.meta_data[pos->second.meta_data.size()-1], 1 ) );
          }
        }
        VELOC_SAFE_CALL( VELOC_Restart_begin( lab.c_str(), current_version ) );
        VELOC_SAFE_CALL( VELOC_Recover_selective(VELOC_RECOVER_SOME, incremental_ids.data(), incremental_ids.size()) );
        status = true;
        VELOC_SAFE_CALL( VELOC_Restart_end( status ) );
        VELOC_SAFE_CALL( VELOC_Restart_begin( lab.c_str(), current_version ) );
        VELOC_SAFE_CALL( VELOC_Recover_selective(VELOC_RECOVER_SOME, metadata_ids.data(), metadata_ids.size()) );
        status = true;
        VELOC_SAFE_CALL( VELOC_Restart_end( status ) );
      
        // Check if we need to copy any views from the incremental store to the view
        for(size_t i=0; i<incremental_indices.size(); i++)
        {
          auto&& view = _views[incremental_indices[i]];
          auto vl = get_canonical_label( view->label() );
          auto pos = m_registry.find( vl );
          if ( pos != m_registry.end() )
          {
            std::cout << pos->second.meta_data[pos->second.meta_data.size()-2] << " Changes, " << pos->second.meta_data[pos->second.meta_data.size()-1] << " Bytes" << std::endl;
            int num_pages = view->span()*view->data_type_size()/page_size;
            if(num_pages*page_size < view->span()*view->data_type_size())
              num_pages+=1;
            
            uint32_t num_changes = pos->second.meta_data[pos->second.meta_data.size()-2];
            for( int j=0; j<num_changes; j++ )
            {
              uint32_t block = -1;
              memcpy(&block, pos->second.incr_buff.data()+j*(page_size+sizeof(uint32_t)), sizeof(uint32_t));
              if(pos->second.changed_buff[block])
              {
                size_t num_write = std::min(page_size, view->span()*view->data_type_size() - page_size*block);
                memcpy((unsigned char*)(view->data())+block*page_size, 
                        pos->second.incr_buff.data()+j*(page_size+sizeof(uint32_t))+sizeof(uint32_t), num_write);
                pos->second.changed_buff[block] = false;
                //std::cout << "restored block " << block << " from checkpoint " << vl << std::endl;
              }
            }
            std::cout << num_changes << " changes restored from " << lab << " version " << current_version << std::endl;
          }
        }
        current_version -= 1;
      }
    }
#endif
  }

  void
  VeloCMemoryBackend::reset()
  {
    std::cout << "Reset backend\n";
    for ( auto &&vr : m_registry )
    {
      VELOC_Mem_unprotect( vr.second.id );
      if( vr.second.incremental )
      {
        VELOC_Mem_unprotect( vr.second.incremental );
      }
    }

    m_registry.clear();

    m_latest_version.clear();
    m_alias_map.clear();
  }
  
  void
  VeloCMemoryBackend::register_hashes( const std::vector< std::unique_ptr< Kokkos::ViewHolderBase > > &views,
                                       const std::vector< Detail::CrefImpl > &crefs  )
  {
    // Clear protected bits
    for ( auto &&p : m_registry )
    {
      p.second.protect = false;
    }

    size_t page_size = sysconf(_SC_PAGE_SIZE);
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

        if ( !view->is_hostspace() || !view->span_is_contiguous() )
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
#ifdef KR_ENABLE_INCREMENTAL_SCAN
      if ((p.first.compare("GDV for graph2") == 0) || (p.first.compare("GDV for graph1") == 0))
      {
        p.second.incremental = true;
        if(p.second.meta_data_id == -1)
          p.second.meta_data_id = ++m_last_id;
        if(p.second.indx_id == -1)
          p.second.indx_id = ++m_last_id;
        if(p.second.indx_buff.size() != p.second.size*p.second.element_size)
        {
          p.second.incr_buff.assign(p.second.size*p.second.element_size, 0x00);
          p.second.indx_buff.assign(p.second.size, 0x00000000);
          p.second.changed_buff.assign(p.second.size, false);
        }
        //std::cout << "View size: " << p.second.size*p.second.element_size << ", # of hashes: " << p.second.meta_data.size()-2 << ", size of buffer: " << p.second.incr_buff.size() << std::endl;
        if( p.second.protect && !p.second.registered )
        {
          std::cout << "Protecting memory id " << p.second.meta_data_id << " with label " << p.first << " meta data " << '\n';
          VELOC_SAFE_CALL( VELOC_Mem_protect( p.second.meta_data_id,&(p.second.num_changes), 1, sizeof(uint32_t) ) );
          std::cout << "Protecting memory id " << p.second.indx_id << " with label " << p.first << " indices\n";
          VELOC_SAFE_CALL( VELOC_Mem_protect( p.second.indx_id, p.second.indx_buff.data(), p.second.indx_buff.size(), sizeof(uint32_t) ) );
          std::cout << "Protecting memory id " << p.second.id << " with label " << p.first << '\n';
          VELOC_SAFE_CALL( VELOC_Mem_protect( p.second.id, p.second.incr_buff.data(), p.second.incr_buff.size(), sizeof(unsigned char) ) );
        }
      }
#endif
#ifdef KR_ENABLE_INCREMENTAL_HASH
      if ((p.first.compare("GDV for graph2") == 0) || (p.first.compare("GDV for graph1") == 0))
      {
        p.second.incremental = true;
        p.second.meta_data_id = ++m_last_id;
        int num_pages = p.second.size*p.second.element_size/page_size;
        if(num_pages*page_size < p.second.size*p.second.element_size)
          num_pages+=1;
        unsigned int new_size = num_pages*(page_size+4);
        //std::cout << "Incremental buffer size: " << new_size << std::endl;
        if(p.second.meta_data.size()-2 != num_pages)
        {
          p.second.meta_data.assign(num_pages+2, 0);
          p.second.incr_buff.assign(new_size, 0x00);
          p.second.changed_buff.assign(num_pages, false);
        }
        //std::cout << "View size: " << p.second.size*p.second.element_size << ", # of hashes: " << p.second.meta_data.size()-2 << ", size of buffer: " << p.second.incr_buff.size() << std::endl;
        if( p.second.protect && !p.second.registered )
        {
          std::cout << "Protecting memory id " << p.second.meta_data_id << " with label " << p.first << " meta data " << '\n';
          VELOC_SAFE_CALL( VELOC_Mem_protect( p.second.meta_data_id, p.second.meta_data.data(), p.second.meta_data.size(), sizeof(uint32_t) ) );
          std::cout << "Protecting memory id " << p.second.id << " with label " << p.first << '\n';
          VELOC_SAFE_CALL( VELOC_Mem_protect( p.second.id, p.second.incr_buff.data(), p.second.incr_buff.size(), sizeof(unsigned char) ) );
        }
      }
#endif
      if ( p.second.protect )
      {
        if ( !p.second.registered && !p.second.incremental)
        {
          std::cout << "Protecting memory id " << p.second.id << " with label " << p.first << '\n';
          VELOC_SAFE_CALL( VELOC_Mem_protect( p.second.id, p.second.ptr, p.second.size, p.second.element_size ) );
          p.second.registered = true;
        }
      } 
      else  //deregister
      {
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
  VeloCRegisterOnlyBackend::checkpoint( const std::string &label, int version, const std::vector<std::unique_ptr<Kokkos::ViewHolderBase>> &views )
  {
    // No-op, don't do anything
  }

  void
  VeloCRegisterOnlyBackend::restart(const std::string &label, int version, const std::vector<std::unique_ptr<Kokkos::ViewHolderBase>> &views)
  {
    // No-op, don't do anything
  }

  VeloCFileBackend::VeloCFileBackend( MPIContext< VeloCFileBackend > &ctx, MPI_Comm mpi_comm,
                                      const std::string &veloc_config )
    : m_mpi_comm( mpi_comm ), m_context( &ctx )
  {
    VELOC_SAFE_CALL( VELOC_Init( mpi_comm, veloc_config.c_str()));
  }
  
  VeloCFileBackend::~VeloCFileBackend()
  {
    VELOC_Finalize( false );
  }
  
  void
  VeloCFileBackend::checkpoint( const std::string &label, int version,
                                const std::vector< std::unique_ptr< Kokkos::ViewHolderBase > > &views )
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
                                  const std::vector< std::unique_ptr< Kokkos::ViewHolderBase>> &views )
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
