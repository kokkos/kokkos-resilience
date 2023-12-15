#include "Registration.hpp"
#include <string>
#include <locale> //isalnum

namespace KokkosResilience::Detail {
  std::string sanitized_label(std::string label){
    //If character not alphanumeric, can only be an underscore.
    for(char& c : label){
      if(!std::isalnum(c)) c = '_';
    }
    return label;
  }
}
