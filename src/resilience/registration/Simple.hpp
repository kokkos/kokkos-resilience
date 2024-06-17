#include "resilience/registration/Registration.hpp"

namespace KokkosResilience::Detail {
  template <typename MemberType>
  struct SimpleRegistration : public RegistrationBase {
    SimpleRegistration() = delete;
    SimpleRegistration(MemberType& member, const std::string label) 
        : RegistrationBase(name), m_member(member) {}

    const serializer_t serializer() const override{
      return [&, this](std::ostream& stream){
        stream.write((const char*)&m_member, sizeof(this->m_member));
        return stream.good();
      };
    }

    const deserializer_t deserializer() const override{
      return [&, this](std::istream& stream){
        stream.read((char*)&m_member, sizeof(m_member));
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

      return &m_member == &(other->m_member);
    }

  private:
    MemberType& m_member;
  };
}

namespace KokkosResilience {
  template<typename Unused, typename T>
  struct create_registration<T, Unused> {
    std::shared_ptr<Detail::SimpleRegistration<T>> reg;

    create_registration(ContextBase& ctx, T& member, std::string label)
        : reg(member, label) {};

    auto get() {
      return std::move(reg);
    }
  };
}
