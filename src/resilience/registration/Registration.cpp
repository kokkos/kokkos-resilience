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

    const bool RegistrationBase::operator==(const RegistrationBase& other) const {
        return this->name == other.name;
    }

    virtual const size_t RegistrationBase::hash() const {
      const size_t base = 7;
      size_t hash = 0;
      for(size_t i = 0; i < name.length(); i++){
        hash += static_cast<size_t>((static_cast<size_t>(name[i]) *
                                     static_cast<size_t>(pow(base, i))
                                    ) % INT_MAX);
      }
      return static_cast<size_t>(hash%INT_MAX);
    }
  }
  
  const size_t Registration::hash() const {
    return (*this)->hash();
  }

  const bool Registration::operator==(const Registration& other) const {
    return *(this->get()) == *(other.get()); 
  }
  
}
