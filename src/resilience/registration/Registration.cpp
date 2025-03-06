#include "Registration.hpp"
#include <string>
#include <locale> //isalnum

namespace KokkosResilience {
  namespace Detail { 
    std::string sanitized_label(std::string label){
      //If character not alphanumeric, can only be an underscore.
      for(char& c : label){
        if(!std::isalnum(c)) c = '_';
      }
      return label;
    }

    size_t label_hash(const std::string& name) {
      const size_t base = 7;
      size_t hash = 0;
      for(size_t i = 0; i < name.length(); i++){
        hash += static_cast<size_t>((static_cast<size_t>(name[i]) *
                                     static_cast<size_t>(pow(base, i))
                                    ) % INT_MAX);
      }
      return static_cast<size_t>(hash%INT_MAX);
    }
   
    RegistrationBase::RegistrationBase(const std::string member_name) :
      name(sanitized_label(member_name)) {
    }

    const bool RegistrationBase::operator==(const RegistrationBase& other) const {
        return this->name == other.name;
    }

    const size_t RegistrationBase::hash() const {
      return label_hash(name);
    }

    const bool RegistrationBase::serialize(std::ostream& out){
      return this->serializer()(out);
    }

    const bool RegistrationBase::deserialize(std::istream& in){
      return this->deserializer()(in);
    }
  }
  
  Registration::Registration(
      serializer_t&& s_fun,
      deserializer_t&& d_fun,
      const std::string& label
  ) : Registration(std::make_shared<Detail::CustomRegistration>(std::move(s_fun), std::move(d_fun), label))
  { }

  const size_t Registration::hash() const {
    return base->hash();
  }

  const bool Registration::operator==(const Registration& other) const {
    return *base.get() == *other.base.get(); 
  }
  
}
