#include "Registration.hpp"
#include <string>
#include <locale>

namespace KokkosResilience {
  std::string sanitized_label(std::string label){
    // If character is not alphanumeric, can only be an underscore.
    for(char& c : label){
      if(!std::isalnum(c)) c = '_';
    }
    return label;
  }

  size_t label_hash(const std::string& name) {
    // Hash by summing each character multiplied by some value based on the
    //  position of the character
    // To try to make sure the hash for "ab" is different from "ba", we want
    //  to make our base position-based value larger than the number of
    //  possible characters. We allow [a-zA-Z0-9_] = 63 characters
    // We also want to make sure the character*position value is unique for
    //  all possible characters and positions (as best we can) - so we make
    //  the position component the ith power of a prime number
    // C++ hashes are type size_t, but we want these hashes to be compatible
    //  as IDs for various checkpointing libraries, which are usually integers

    // 67 is first prime number larger than 63
    const size_t base = 67;
    size_t hash = 0;
    for(size_t i = 0; i < name.length(); i++){
      float character_val = name[i];
      float position_val = powf(base, i);
      hash += static_cast<size_t>( fmod(character_val * position_val, INT_MAX) );
    }
    return hash % INT_MAX;
  }
}

namespace KokkosResilience::Impl::RegistrationImpl {
  Base::Base(const std::string member_name) :
    name(sanitized_label(member_name)) {
  }

  const bool Base::operator==(const Base& other) const {
      return this->name == other.name;
  }

  const size_t Base::hash() const {
    return label_hash(name);
  }

  const bool Base::serialize(std::ostream& out){
    return this->serializer()(out);
  }

  const bool Base::deserialize(std::istream& in){
    return this->deserializer()(in);
  }
  
  Registration::Registration(
      serializer_t&& s_fun,
      deserializer_t&& d_fun,
      const std::string& label
  ) : Registration(
        std::make_shared<Custom>(std::move(s_fun), std::move(d_fun), label)
      )
  { }

  const size_t Registration::hash() const {
    return base->hash();
  }

  const bool Registration::operator==(const Registration& other) const {
    return *base.get() == *other.base.get(); 
  }
  
}
