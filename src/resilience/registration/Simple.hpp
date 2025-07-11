#include "resilience/registration/Registration.hpp"

namespace KokkosResilience::RegistrationImpl {
  template <typename MemberType>
  class Simple : public Base {
  public:
    Simple() = delete;
    Simple(ContextBase&, MemberType& member, const std::string name) 
      : Base(name), member_ptr(reinterpret_cast<char *>(&member)) {
      static_assert(std::is_trivially_copyable_v<MemberType>);
    }

    const serializer_t serializer() const override{
      return [&, this](std::ostream& stream){
        stream.write(member_ptr, sizeof(MemberType));
        return stream.good();
      };
    }

    const deserializer_t deserializer() const override{
      return [&, this](std::istream& stream){
        stream.read(member_ptr, sizeof(MemberType));
        return stream.good();
      };
    }
    
    const bool is_same_reference(const Registration& other_reg) const override{
      auto other = dynamic_cast<Simple*>(other_reg.get());
      
      if(!other){
        fprintf(stderr,
          "KokkosResilience: Warning, member name %s is shared by more than 1"
          " registration type\n", name.c_str()
        );
        return false;
      }

      return member_ptr == other->member_ptr;
    }

  private:
    char* member_ptr;
  };
}
