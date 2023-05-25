#include "Automatic.hpp"
#include <stdexcept>

namespace KokkosResilience::Detail {
  AutomaticBackend make_backend(ContextBase* ctx){
    auto backend = ctx->config()["backend"].as<std::string>();

#ifdef KR_ENABLE_VELOC
    if(backend == "veloc"){
      return std::make_shared<VeloCMemoryBackend>(ctx);
    }
#endif
      
    throw std::runtime_error(backend + " backend is not available"); 
  }
}
