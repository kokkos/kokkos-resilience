#include "resilience/registration/Registration.impl.hpp"

namespace KokkosResilience {
  Registration custom_registration(
      serializer_t&& s_fun,
      deserializer_t&& d_fun,
      const std::string label){
    return std::make_shared<Detail::CustomRegistration>(std::move(s_fun), std::move(d_fun), label);
  }
}
