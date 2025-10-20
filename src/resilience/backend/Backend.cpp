#include "resilience/context/Context.hpp"
#include <stdexcept>

#include "resilience/backend/StdFile.hpp"

#ifdef KR_ENABLE_VELOC_BACKEND
#include "resilience/backend/Veloc.hpp"
#endif

namespace KokkosResilience::Impl {
  Backend make_backend(ContextBase& ctx){
    auto backend = ctx.config()["backend"].as<std::string>();

    if(backend == "stdfile"){
      return std::make_shared<BackendImpl::StdFile>(ctx);
    }
      
#ifdef KR_ENABLE_VELOC_BACKEND
    if(backend == "veloc"){
      return std::make_shared<BackendImpl::VeloC>(ctx);
    }
#endif

    throw std::runtime_error(backend + " backend is not available"); 
  }
}
