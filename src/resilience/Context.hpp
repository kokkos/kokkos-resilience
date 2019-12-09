#ifndef INC_RESILIENCE_CONTEXT_HPP
#define INC_RESILIENCE_CONTEXT_HPP

#include <string>
#include <utility>
#include <memory>
#include <functional>
#ifdef KR_ENABLE_VELOC
#include <mpi.h>
#endif
#include "Config.hpp"
#include "Cref.hpp"
#include "CheckpointFilter.hpp"
#include <Kokkos_Core.hpp>
#include <Kokkos_ViewHooks.hpp>

// Tracing support
#ifdef KR_ENABLE_TRACING
#include "util/Trace.hpp"
#endif

namespace KokkosResilience
{
  namespace detail
  {
  }

  class ContextBase
  {
  public:

    explicit ContextBase( Config cfg );

    virtual ~ContextBase() = default;

    virtual void register_hashes(const std::vector< std::unique_ptr< Kokkos::ViewHolderBase > > &views,
                                 const std::vector< Detail::CrefImpl > &crefs) = 0;
    virtual bool restart_available( const std::string &label, int version ) = 0;
    virtual void restart( const std::string &label, int version,
                          const std::vector< std::unique_ptr< Kokkos::ViewHolderBase > > &views ) = 0;
    virtual void checkpoint( const std::string &label, int version,
                             const std::vector< std::unique_ptr< Kokkos::ViewHolderBase > > &views ) = 0;

    virtual void reset() = 0;

    std::function< bool( int ) > default_filter() const noexcept { return m_default_filter; }

    Config &config() noexcept { return m_config; }
    const Config &config() const noexcept { return m_config; }

  private:

    Config m_config;

    std::function< bool( int ) > m_default_filter;
  };
  
  template< typename Backend >
  class Context : public ContextBase
  {
  public:
    
    explicit Context( MPI_Comm comm, Config &cfg )
      : ContextBase( cfg ), m_backend( *this, comm ), m_comm( comm )
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
  
    Backend &backend() { return m_backend; }


    void register_hashes( const std::vector< std::unique_ptr< Kokkos::ViewHolderBase > > &views,
                          const std::vector< Detail::CrefImpl > &crefs ) override
    {
      m_backend.register_hashes( views, crefs );
    }

    bool restart_available( const std::string &label, int version ) override
    {
      return m_backend.restart_available( label, version );
    }

    void restart( const std::string &label, int version,
                  const std::vector< std::unique_ptr< Kokkos::ViewHolderBase > > &views ) override
    {
      m_backend.restart( label, version, views );
    }

    void checkpoint( const std::string &label, int version,
                     const std::vector< std::unique_ptr< Kokkos::ViewHolderBase > > &views ) override
    {
      m_backend.checkpoint( label, version, views );
    }

    void reset() override
    {
      m_backend.reset();
    }
    
#ifdef KR_ENABLE_TRACING
    Util::detail::TraceStack  &trace() { return m_trace; };
#endif
  
  private:
    
    MPI_Comm  m_comm;
    Backend m_backend;
    
#ifdef KR_ENABLE_TRACING
    Util::detail::TraceStack  m_trace;
#endif
  };

  std::unique_ptr< ContextBase > make_context( MPI_Comm comm, const std::string &config );
}

#endif  // INC_RESILIENCE_CONTEXT_HPP
