#include "veloc_backend.hpp"

#include <sstream>
#include <fstream>
#include <veloc.h>

#include "../context.hpp"
#include "../util/trace.hpp"

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
  
  VeloCCheckpointBackend::VeloCCheckpointBackend( Context< VeloCCheckpointBackend > &ctx, MPI_Comm mpi_comm, const std::string &veloc_config )
    : m_mpi_comm( mpi_comm ), m_context( &ctx )
  {
    VELOC_SAFE_CALL( VELOC_Init( mpi_comm, veloc_config.c_str()));
  }
  
  VeloCCheckpointBackend::~VeloCCheckpointBackend()
  {
    VELOC_Finalize( false );
  }
  
  void
  VeloCCheckpointBackend::checkpoint( const std::string &label, int version,
                                      const std::vector< std::unique_ptr< Kokkos::ViewHolderBase > > &views )
  {
    // Wait for previous checkpoint to finish
    VELOC_SAFE_CALL( VELOC_Checkpoint_wait() );
    
    // Start new checkpoint
    VELOC_SAFE_CALL( VELOC_Checkpoint_begin( label.c_str(), version ) );
    
    char veloc_file_name[VELOC_MAX_NAME];
    
    bool status = true;
    try
    {
      VELOC_SAFE_CALL( VELOC_Route_file( veloc_file_name ) );
      printf( "veloc file name: %s\n", veloc_file_name );
      
      std::string fname( veloc_file_name );
      std::ofstream vfile( fname, std::ios::binary );

#ifdef KR_ENABLE_TRACING
      auto write_trace = Util::begin_trace< Util::TimingTrace< std::string > >( *m_context, "write" );
#endif
      for ( auto &&v : views )
      {
        char *bytes = static_cast< char * >( v->data() );
        std::size_t len = v->span();
        
        vfile.write( bytes, len );
      }
#ifdef KR_ENABLE_TRACING
      write_trace.end();
#endif
    } catch ( ... ) {
      status = false;
    }
    
    VELOC_SAFE_CALL( VELOC_Checkpoint_end( status ) );
  }
  
  bool
  VeloCCheckpointBackend::restart_available( const std::string &label, int version )
  {
    int latest = VELOC_Restart_test( label.c_str(), 0 );
 
    // res is < 0 if no versions available, else it is the latest version
    return version <= latest;
  }

  int VeloCCheckpointBackend::latest_version( const std::string &label ) {
      return VELOC_Restart_test(label.c_str(), 0);
   }
  
  void VeloCCheckpointBackend::restart( const std::string &label, int version,
                                        const std::vector< std::unique_ptr< Kokkos::ViewHolderBase>> &views )
  {
    VELOC_SAFE_CALL( VELOC_Restart_begin( label.c_str(), version ) );
    
    char veloc_file_name[VELOC_MAX_NAME];
    
    bool status = true;
    try
    {
      VELOC_SAFE_CALL( VELOC_Route_file( veloc_file_name ) );
      printf( "restore file name: %s\n", veloc_file_name );
      
      std::string fname( veloc_file_name );
      std::ifstream vfile( fname, std::ios::binary );

#ifdef KR_ENABLE_TRACING
      auto read_trace = Util::begin_trace< Util::TimingTrace< std::string > >( *m_context, "read" );
#endif
      for ( auto &&v : views )
      {
        char *bytes = static_cast< char * >( v->data() );
        std::size_t len = v->span();
        
        vfile.read( bytes, len );
      }
#ifdef KR_ENABLE_TRACING
      read_trace.end();
#endif
    } catch ( ... ) {
      status = false;
    }
    
    VELOC_SAFE_CALL( VELOC_Restart_end( status ) );
  }
}
