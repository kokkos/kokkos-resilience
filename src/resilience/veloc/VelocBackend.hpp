#ifndef INC_RESILIENCE_VELOC_VELOCBACKEND_HPP
#define INC_RESILIENCE_VELOC_VELOCBACKEND_HPP

#include <string>
#include <vector>
#include <memory>
#include <Kokkos_Core.hpp>
#include <Kokkos_ViewHooks.hpp>
#include <unordered_map>
#include <unordered_set>
#include <mpi.h>

namespace KokkosResilience
{
  namespace detail
  {
    struct cref_impl;
  }
  
  template< typename Backend >
  class Context;
  
  class VeloCMemoryBackend
  {
  public:
    
    using context_type = Context< VeloCMemoryBackend >;
    
    VeloCMemoryBackend( context_type &ctx, MPI_Comm mpi_comm, const std::string &veloc_config );
    ~VeloCMemoryBackend();
  
    void checkpoint( const std::string &label, int version,
                     const std::vector< std::unique_ptr< Kokkos::ViewHolderBase > > &views );
  
    bool restart_available( const std::string &label, int version );
    int latest_version (const std::string &label);
  
    void restart( const std::string &label, int version,
                  const std::vector< std::unique_ptr< Kokkos::ViewHolderBase > > &views );
  
    void register_hashes( const std::vector< std::unique_ptr< Kokkos::ViewHolderBase > > &views,
      const std::vector< detail::cref_impl > &crefs );

    void reset();

  private:
    
    std::unordered_set< void * > m_cref_registry;
    std::unordered_map< std::uintptr_t, std::vector< unsigned char > > m_view_registry;
    
    MPI_Comm m_mpi_comm;
    context_type *m_context;
    
    int m_latest_version;
  };
  
  class VeloCFileBackend
  {
  public:
  
    VeloCFileBackend( Context< VeloCFileBackend > &ctx, MPI_Comm mpi_comm, const std::string &veloc_config);
    ~VeloCFileBackend();
  
    void checkpoint( const std::string &label, int version,
                     const std::vector< std::unique_ptr< Kokkos::ViewHolderBase > > &views );
  
    bool restart_available( const std::string &label, int version );
    int latest_version (const std::string &label);
  
    void restart( const std::string &label, int version,
                  const std::vector< std::unique_ptr< Kokkos::ViewHolderBase > > &views );
  
    void register_hashes( const std::vector< std::unique_ptr< Kokkos::ViewHolderBase > > & ) {} // Do nothing

  private:
  
    MPI_Comm m_mpi_comm;
    Context< VeloCFileBackend > *m_context;
  };
}

#endif  // INC_RESILIENCE_VELOC_VELOCBACKEND_HPP
