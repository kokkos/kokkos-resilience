#ifndef INC_RESILIENCE_MAGISTRATE_HPP
#define INC_RESILIENCE_MAGISTRATE_HPP

#ifdef KR_ENABLE_MAGISTRATE

#include "resilience/registration/Registration.hpp"
#include "resilience/view_hooks/ViewHolder.hpp"
#include <checkpoint/checkpoint.h>
#include <checkpoint/serializers/stream_serializer.h>

namespace KokkosResilience {
  class ContextBase;
}

namespace KokkosResilience::Detail {
  struct Checkpoint_Trait {};

  //Registration for some type which Magistrate knows how to checkpoint.
  template
  <
    typename MemberType,
    typename... Traits
  >
  struct MagistrateRegistration : public RegistrationBase {
    MagistrateRegistration() = delete;

    MagistrateRegistration(MemberType& member, std::string name)
      : RegistrationBase(name), m_member(member) {}

    const serializer_t serializer() const override{
      return [&, this](std::ostream& stream){
        checkpoint::serializeToStream<
          Traits...
        >(m_member, stream);
        return stream.good();
      };
    }

    const deserializer_t deserializer() const override{
      return [&, this](std::istream& stream){
        checkpoint::deserializeInPlaceFromStream<
          Traits...
        >(stream, &m_member);
        return stream.good();
      };
    }

    const bool is_same_reference(const Registration& other_reg) const override{
      auto other = dynamic_cast<MagistrateRegistration*>(other_reg.get());

      if(!other){
        //We wouldn't expect this to happen, and it may indicate a hash collision
        fprintf(stderr, "KokkosResilience: Warning, member name %s is shared by more than 1 registration type\n", name.c_str());
        return false;
      }

      return &m_member == &other->m_member;
    }

  private:
    MemberType& m_member;
  };
}

namespace KokkosResilience {
  template<
    typename T,
    typename... Traits
  >
  struct create_registration<
    T,
    std::tuple<Traits...>,
    std::enable_if_t<
      checkpoint::SerializableTraits<T, checkpoint::StreamPacker<>>::is_traversable
    >*
  > {
    using BaseT = Detail::MagistrateRegistration<T, Traits...>;
    std::shared_ptr<BaseT> reg;

    create_registration(ContextBase& ctx, T& member, std::string label)
        : reg(std::make_shared<BaseT>(member, label)) {};

    auto get() && {
      return std::move(reg);
    }
  };
}


#endif

#endif  // INC_RESILIENCE_MAGISTRATE_HPP
