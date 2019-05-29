#ifndef INC_RESILIENCE_CONTEXT_HPP
#define INC_RESILIENCE_CONTEXT_HPP

#include <utility>
#include <mpi.h>

#include "veloc/veloc_backend.hpp"

namespace KokkosResilience
{
  template< typename Backend >
  class Context;
  
  template<>
  class Context< VeloCCheckpointBackend >
  {
  public:
    
    explicit Context( MPI_Comm comm, const std::string &config )
      : m_backend( comm, config ), m_comm( comm )
    {
    
    }
    
    ~Context()
    {
    
    }
    
    MPI_Comm comm() const noexcept { return m_comm; }
    
    VeloCCheckpointBackend &backend() { return m_backend; }
    
  private:
    
    MPI_Comm  m_comm;
    VeloCCheckpointBackend m_backend;
  };
}

#endif  // INC_RESILIENCE_CONTEXT_HPP
