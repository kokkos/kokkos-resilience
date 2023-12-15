#include "Automatic.hpp"
#include "resilience/context/ContextBase.hpp"
#include <stdexcept>

#ifdef KR_ENABLE_VELOC
#include "veloc/VelocBackend.hpp"
#endif

#ifdef KR_ENABLE_STDFILE
#include "stdfile/StdFileBackend.hpp"
#endif

namespace KokkosResilience::Detail {
  AutomaticBackend make_backend(ContextBase& ctx){
    auto backend = ctx.config()["backend"].as<std::string>();

#ifdef KR_ENABLE_VELOC
    if(backend == "veloc"){
      return std::make_shared<VeloCMemoryBackend>(ctx);
    }
#endif
#ifdef KR_ENABLE_STDFILE
    if(backend == "stdfile"){
      return std::make_shared<StdFileBackend>(ctx);
    }
#endif
      
    throw std::runtime_error(backend + " backend is not available"); 
  }
}
