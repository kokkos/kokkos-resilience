#include "VelocBackend.hpp"

#include <sstream>
#include <fstream>
#include <veloc.h>
#include <cassert>

#include "../MPIContext.hpp"
#include "../AutomaticCheckpoint.hpp"

#ifdef KR_ENABLE_TRACING
   #include "../util/Trace.hpp"
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
  
  VeloCMemoryBackend::VeloCMemoryBackend( context_type &ctx, MPI_Comm mpi_comm )
    : m_mpi_comm( mpi_comm ), m_context( &ctx ), m_latest_version( -2 )
  {
    const auto &vconf = m_context->config()["backends"]["veloc"]["config"].as< std::string >();
    VELOC_SAFE_CALL( VELOC_Init( mpi_comm, vconf.c_str() ) );
  }
  
  VeloCMemoryBackend::~VeloCMemoryBackend()
  {
    VELOC_Checkpoint_wait();
    VELOC_Finalize( false );
  }
  
  void VeloCMemoryBackend::checkpoint( const std::string &label, int version,
                                       const std::vector< std::unique_ptr< Kokkos::ViewHolderBase > > &_views )
  {
    bool status = true;
    
    // Check if we need to copy any views to backing store
    for ( auto &&view : _views )
    {
      if ( !view->span_is_contiguous() || !view->is_hostspace() )
      {
        auto pos = m_view_registry.find( Detail::MemProtectKey{ view->data() } );
        if ( pos != m_view_registry.end())
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
  
    m_latest_version = version;
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
    if ( m_latest_version < -1 )
      m_latest_version = VELOC_Restart_test( label.c_str(), 0 );
    
    return m_latest_version;
  }
  
  void
  VeloCMemoryBackend::restart( const std::string &label, int version,
    const std::vector< std::unique_ptr< Kokkos::ViewHolderBase > > &_views )
  {
    VELOC_SAFE_CALL( VELOC_Restart_begin( label.c_str(), version ));
    
    bool status = true;
    
    VELOC_SAFE_CALL( VELOC_Recover_mem() );
    
    VELOC_SAFE_CALL( VELOC_Restart_end( status ) );
  
    // Check if we need to copy any views from the backing store back to the view
    for ( auto &&view : _views )
    {
      if ( !view->span_is_contiguous() || !view->is_hostspace() )
      {
        auto pos = m_view_registry.find( Detail::MemProtectKey{ view->data() } );
        if ( pos != m_view_registry.end() )
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
    for ( auto &&vr : m_view_registry )
    {
      VELOC_Mem_unprotect( vr.second.id );
    }

    for ( auto &&cr : m_cref_registry )
    {
      VELOC_Mem_unprotect( cr.second.id );
    }

    m_view_registry.clear();
    m_cref_registry.clear();

    m_latest_version = -2;
  }
  
  void
  VeloCMemoryBackend::register_hashes( const std::vector< std::unique_ptr< Kokkos::ViewHolderBase > > &views,
                                       const std::vector< Detail::CrefImpl > &crefs  )
  {
    for ( auto &&view : views )
    {
      if ( !view->data() )  // uninitialized view
        continue;
      // If we haven't already register, register with VeloC
      if ( m_view_registry.find( Detail::MemProtectKey{ view->data() } ) == m_view_registry.end() )
      {
        int id = static_cast< int >( m_view_registry.size() + m_cref_registry.size() );
        auto type_size = view->data_type_size();
        auto span = view->span();
        
        std::vector< unsigned char > buff;
        
        if ( !view->is_hostspace() || !view->span_is_contiguous() )
        {
          // Can't reference memory directly, allocate memory for a watch buffer
          buff.assign( span * type_size, 0x00 );
          VELOC_SAFE_CALL( VELOC_Mem_protect( id, buff.data(), span, type_size ) );
        } else {
          VELOC_SAFE_CALL( VELOC_Mem_protect( id, view->data(), span, type_size ) );
        }
  
        m_view_registry.emplace( std::piecewise_construct,
          std::forward_as_tuple( view->data() ),
          std::forward_as_tuple( id, std::move( buff ) ) );
      }
    }
    
    // Register crefs
    for ( auto &&cref : crefs )
    {
      if ( !cref.ptr )  // uninitialized view
        continue;
      // If we haven't already register, register with VeloC
      if ( m_cref_registry.find( Detail::MemProtectKey{ cref.ptr } ) == m_cref_registry.end())
      {
        int id = static_cast< int >( m_view_registry.size() + m_cref_registry.size());
  
        VELOC_SAFE_CALL( VELOC_Mem_protect( id, cref.ptr, cref.num, cref.sz ) );
        
        m_cref_registry.emplace( std::piecewise_construct,
            std::forward_as_tuple( cref.ptr ),
            std::forward_as_tuple( id ) );
      }
    }
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
