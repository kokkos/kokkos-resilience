#include "Registration.impl.hpp"

namespace KokkosResilience::Detail {
  struct CustomRegistration : public RegistrationBase {
    CustomRegistration() = delete;
    CustomRegistration(serializer_t&& serializer, deserializer_t&& deserializer, const std::string name) : 
        RegistrationBase(name),
        m_serializer(serializer),
        m_deserializer(deserializer) {};

    const serializer_t serializer() const override{
        return m_serializer;
    }

    const deserializer_t deserializer() const override{
        return m_deserializer;
    }

    const bool is_same_reference(const Registration& other_reg) const override{
      auto other = dynamic_cast<CustomRegistration*>(other_reg.get());
      
      if(!other){
        //We wouldn't expect this to happen, and it may indicate a hash collision
        fprintf(stderr, "KokkosResilience: Warning, member name %s is shared by more than 1 registration type\n", name.c_str());
        return false;
      }

      return (&m_serializer == &(other->m_serializer)) && 
           (&m_deserializer == &(other->m_deserializer));
    }

  private:
    const serializer_t m_serializer;
    const deserializer_t m_deserializer;
  };
}
