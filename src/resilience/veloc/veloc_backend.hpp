#ifndef INC_RESILIENCE_VELOC_BACKEND_HPP
#define INC_RESILIENCE_VELOC_BACKEND_HPP

#include <string>
#include <vector>
#include <memory>
#include <Kokkos_Core.hpp>
#include <Kokkos_ViewHooks.hpp>
#include <mpi.h>

namespace KokkosResilience
{
  template< typename Backend >
  class Context;
  
  class VeloCCheckpointBackend
  {
  public:
  
    VeloCCheckpointBackend( Context< VeloCCheckpointBackend > &ctx, MPI_Comm mpi_comm, const std::string &veloc_config);
    ~VeloCCheckpointBackend();
  
    void checkpoint( const std::string &label, int version,
                     const std::vector< std::unique_ptr< Kokkos::ViewHolderBase > > &views );
  
    bool restart_available( const std::string &label, int version );
    int latest_version (const std::string &label);
  
    void restart( const std::string &label, int version,
                  const std::vector< std::unique_ptr< Kokkos::ViewHolderBase > > &views );

  private:
  
    MPI_Comm m_mpi_comm;
    Context< VeloCCheckpointBackend > *m_context;
  };
}

#endif  // INC_RESILIENCE_VELOC_BACKEND_HPP
