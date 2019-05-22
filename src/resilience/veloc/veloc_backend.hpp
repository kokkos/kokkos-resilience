#ifndef INC_RESILIENCE_VELOC_BACKEND_HPP
#define INC_RESILIENCE_VELOC_BACKEND_HPP

#include <string>
#include <vector>
#include <memory>
#include <Kokkos_ViewHooks.hpp>

namespace KokkosResilience
{
  class VeloCCheckpointBackend
  {
  public:
  
    VeloCCheckpointBackend( int mpi_comm, const std::string &veloc_config);
    ~VeloCCheckpointBackend();
  
    void checkpoint( const std::string &label, int version,
                     const std::vector< std::unique_ptr< Kokkos::ViewHolderBase > > &views );
  
    bool restart_available( const std::string &label, int version );
  
    void restart( const std::string &label, int version,
                  const std::vector< std::unique_ptr< Kokkos::ViewHolderBase > > &views );

  private:
  
    int m_mpi_comm;
  };
}

#endif  // INC_RESILIENCE_VELOC_BACKEND_HPP
