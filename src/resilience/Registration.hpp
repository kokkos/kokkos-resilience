#ifndef _INC_RESILIENCE_REGISTRATION_HPP
#define _INC_RESILIENCE_REGISTRATION_HPP

#include <string>
#include <functional>
#include "view_hooks/ViewHolder.hpp"

#ifdef KR_ENABLE_MAGISTRATE
#include "checkpoint/checkpoint.h"
#endif

namespace KokkosResilience
{
  typedef std::function<bool (std::ostream &)> serializer_t;
  typedef std::function<bool (std::istream &)> deserializer_t;

  struct RegistrationBase;
  typedef std::shared_ptr<RegistrationBase> Registration;

  struct RegistrationBase {
    const std::string name;

    RegistrationBase() = delete;
    ~RegistrationBase() = default;

    virtual const serializer_t serializer() const = 0;
    virtual const deserializer_t deserializer() const = 0;
    virtual const bool is_same_reference(const Registration&) const = 0;
    
    bool operator==(const RegistrationBase& other) const {
        return this->name == other.name;
    }

  protected:
    RegistrationBase(const std::string& member_name) : 
        name(member_name) { }
  };

} //namespace KokkosResilience

namespace std {
  template<>
  struct hash<KokkosResilience::Registration>{
    std::size_t operator()(const KokkosResilience::Registration& registration) const {
      const std::size_t base = 31;
      std::size_t hash = 0;
      for(size_t i = 0; i < registration->name.length(); i++){
        hash += static_cast<std::size_t>((static_cast<std::size_t>(registration->name[i]) * static_cast<std::size_t>(pow(base, i))) % INT_MAX);
      }
      return static_cast<std::size_t>(hash%INT_MAX);
    }
  };

  bool operator==(const KokkosResilience::Registration& lhs, const KokkosResilience::Registration& rhs);
}

namespace KokkosResilience
{
  namespace Detail 
  {
    template<typename Context>
    struct ViewHolderRegistration : public RegistrationBase {
      ViewHolderRegistration() = delete;
      ViewHolderRegistration(const KokkosResilience::ViewHolder& view, Context& ctx) : RegistrationBase(view->label()), m_view(view), m_ctx(ctx) {};

      const serializer_t serializer() const override{
#ifdef KR_ENABLE_MAGISTRATE
        return [&, this](std::ostream& stream){
            m_view->serialize(stream);
            return stream.good();
        };
#else
        if(!m_view->span_is_contiguous() || !m_view->is_host_space()){
          std::vector<char> buf = m_ctx.get_buf(m_view->data_type_size() * m_view->span());
          return [&, this](std::ostream& stream){
              m_view->serialize(stream, buf.data());
              return stream.good();
          };
        } else {
          return [&, this](std::ostream& stream){
              m_view->serialize(stream);
              return stream.good();
          };
        }
#endif
      }

      const deserializer_t deserializer() const override{
#ifdef KR_ENABLE_MAGISTRATE 
        return [&, this](std::istream& stream){
            m_view->deserialize(stream);
            return stream.good();
        };
#else
        if(!m_view->span_is_contiguous() || !m_view->is_host_space()){
          std::vector<char> buf = m_ctx.get_buf(m_view->data_type_size() * m_view->span());
          return [&, this](std::istream& stream) {
              m_view->deserialize(stream, buf.data());
              return stream.good();
          };
        } else {
          return [&, this](std::istream& stream){
              m_view->deserialize(stream);
              return stream.good();
          };
        }
#endif
      }
    
      const bool is_same_reference(const Registration& other_reg) const override{
        auto other = std::dynamic_pointer_cast<ViewHolderRegistration>(other_reg);
        
        if(!other){
          //We wouldn't expect this to happen, and it may indicate a hash collision
          fprintf(stderr, "KokkosResilience: Warning, member name %s is shared by more than 1 registration type\n", name.c_str());
          return false;
        }
      
        //Handle subviews! We want to checkpoint the largest view/subview, so report that the other is 
        //the same reference if they're a subset of me.
        //
        //TODO: This currently assumes the two views are equal or subviews (ie no name collisions),
        //      and that a larger data() pointer implies a subview (ie we can deal well with subviews of 
        //      subviews, but not two different subviews of the same view). Does Kokkos expose anything
        //      that can help with this?
        return m_view->data() <= other->m_view->data();
      }
    private:      
      const KokkosResilience::ViewHolder m_view;
      Context& m_ctx;
    };
      

#ifdef KR_ENABLE_MAGISTRATE
    template <typename MemberType, std::enable_if_t<checkpoint::SerializableTraits<MemberType>::is_traversable>* = nullptr>
    struct MagistrateRegistration : public RegistrationBase {
      MagistrateRegistration() = delete;
      
      MagistrateRegistration(const std::string& name, MemberType& member) : RegistrationBase(name), m_member(member) {}

      const serializer_t serializer() const override{
        return [&, this](std::ostream& stream){
          checkpoint::serializeToStream(m_member, stream);
          return stream.good();
        };
      }

      const deserializer_t deserializer() const override{
        return [&, this](std::istream& stream){
          checkpoint::deserializeInPlaceFromStream<MemberType>(stream, &m_member);
          return stream.good();
        };
      }

      const bool is_same_reference(const Registration& other_reg) const override{
        auto other = std::dynamic_pointer_cast<MagistrateRegistration>(other_reg);
        
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
#endif

    template <typename MemberType>
    struct SimpleRegistration : public RegistrationBase {
      SimpleRegistration() = delete;
      SimpleRegistration(const std::string& name, MemberType& member) : RegistrationBase(name), m_member(member) {}

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
        auto other = std::dynamic_pointer_cast<SimpleRegistration>(other_reg);
        
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

    struct CustomRegistration : public RegistrationBase {
      CustomRegistration() = delete;
      CustomRegistration(const std::string& name, const serializer_t& serializer, const deserializer_t& deserializer) : 
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
        auto other = std::dynamic_pointer_cast<CustomRegistration>(other_reg);
        
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
  } //namespace detail

  template<typename Context>
  inline Registration get_registration(const KokkosResilience::ViewHolder& view, Context& ctx){
    return std::make_shared<Detail::ViewHolderRegistration<Context>>(view, ctx);
  }
  
  template<typename MemberType>
  inline Registration get_registration(const std::string& name, MemberType& member){
#ifdef KR_ENABLE_MAGISTRATE
    if constexpr(checkpoint::SerializableTraits<MemberType>::is_traversable){
      return std::make_shared<Detail::MagistrateRegistration<MemberType>>(name, member);
    } 
#endif
    return std::make_shared<Detail::SimpleRegistration<MemberType>>(name, member);
  }

  inline Registration get_registration(const std::string& name, const serializer_t& serializer, const deserializer_t& deserializer){
    return std::make_shared<Detail::CustomRegistration>(name, serializer, deserializer);
  }

} //namespace KokkosResilience

#endif //_INC_RESILIENCE_REGISTRATION_HPP
