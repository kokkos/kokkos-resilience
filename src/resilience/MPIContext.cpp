#include "MPIContext.hpp"
#ifdef KR_ENABLE_VELOC
#include "veloc/VelocBackend.hpp"
#endif
#include <unordered_map>
#include <functional>

namespace KokkosResilience {
std::unique_ptr< ContextBase >
make_context( MPI_Comm comm, const std::string &config )
{
  auto cfg = Config{ config };

  using fun_type = std::function< std::unique_ptr< ContextBase >() >;
  static std::unordered_map< std::string, fun_type > backends = {
#ifdef KR_ENABLE_VELOC
      { "veloc", [&](){ return std::make_unique< MPIContext< VeloCMemoryBackend > >( comm, cfg ); } }
#endif
  };

  auto pos = backends.find( cfg["backend"].as< std::string >() );
  if ( pos == backends.end() )
    return std::unique_ptr< ContextBase >{};

  return pos->second();
}
}
