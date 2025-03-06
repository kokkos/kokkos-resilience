#include "resilience/registration/Registration.hpp"

namespace KokkosResilience::Detail {
  template <typename MemberType>
  struct SimpleRegistration : public RegistrationBase {
    SimpleRegistration() = delete;
    SimpleRegistration(MemberType& m_member, const std::string m_name) 
        : RegistrationBase(m_name), member(m_member) {}

    const serializer_t serializer() const override{
      return [&, this](std::ostream& stream){
        stream.write((const char*)&member, sizeof(this->member));
        return stream.good();
      };
    }

    const deserializer_t deserializer() const override{
      return [&, this](std::istream& stream){
        stream.read((char*)&member, sizeof(member));
        return stream.good();
      };
    }
    
    const bool is_same_reference(const Registration& other_reg) const override{
      auto other = dynamic_cast<SimpleRegistration*>(other_reg.get());
      
      if(!other){
        //We wouldn't expect this to happen, and it may indicate a hash collision
        fprintf(stderr, "KokkosResilience: Warning, member name %s is shared by more than 1 registration type\n", name.c_str());
        return false;
      }

      return &member == &(other->member);
    }

  private:
    MemberType& member;
  };
}
