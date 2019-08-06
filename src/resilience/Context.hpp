#ifndef INC_RESILIENCE_CONTEXT_HPP
#define INC_RESILIENCE_CONTEXT_HPP

#include <utility>
#ifdef KOKKOS_ENABLE_VELOC
#include <mpi.h>
#endif

#ifdef KOKKOS_ENABLE_VELOC
   #include "veloc/VelocBackend.hpp"
#endif

// Tracing support
#ifdef KR_ENABLE_TRACING
#include "util/trace.hpp"
#endif

namespace KokkosResilience
{
  namespace detail
  {
  }
  
  template< typename Backend >
  class Context;
  
#ifdef KOKKOS_ENABLE_VELOC
  template<>
  class Context< VeloCCheckpointBackend >
  {
  public:
    
    explicit Context( MPI_Comm comm, const std::string &config )
      : m_backend( *this, comm, config ), m_comm( comm )
    {
    
    }
    
    Context( const Context & ) = delete;
    Context( Context && ) = default;
    
    Context &operator=( const Context & ) = delete;
    Context &operator=( Context && ) = default;
    
    ~Context()
    {
#ifdef KR_ENABLE_TRACING
      int rank = -1;
      MPI_Comm_rank( m_comm, &rank );
      int size = -1;
      MPI_Comm_size( m_comm, &size );
  
      std::ostringstream fname;
      fname << "trace" << rank << ".json";
      
      std::ofstream out( fname.str() );
  
      std::cout << "writing trace to " << fname.str() << '\n';
  
      m_trace.write( out );
  
      // Metafile
      picojson::object root;
      root["num_ranks"] = picojson::value( static_cast< double >( size ) );
      
      std::ofstream meta_out( "meta.json" );
      picojson::value( root ).serialize( std::ostream_iterator< char >( meta_out ), true );
#endif
    }
    
    MPI_Comm comm() const noexcept { return m_comm; }
    
    VeloCCheckpointBackend &backend() { return m_backend; }
    
#ifdef KR_ENABLE_TRACING
    Util::detail::TraceStack  &trace() { return m_trace; };
#endif
  
  private:
    
    MPI_Comm  m_comm;
    VeloCCheckpointBackend m_backend;
    
#ifdef KR_ENABLE_TRACING
    Util::detail::TraceStack  m_trace;
#endif
  };

#endif  // KOKKOS_ENABLE_VELOC

}

#endif  // INC_RESILIENCE_CONTEXT_HPP
