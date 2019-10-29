#include "Context.hpp"
#include <fstream>
#include "veloc/VelocBackend.hpp"

namespace KokkosResilience
{
  std::unique_ptr< ContextBase >
  make_context( MPI_Comm comm, const std::string &config )
  {
    auto cfg = Config{ config };

    // Check backend
    if ( cfg["backend"].as< std::string >() == "veloc" )
    {
      return std::make_unique< Context< VeloCMemoryBackend > >( comm, cfg );
    } else {
      return std::unique_ptr< ContextBase >{};
    }

  }
}