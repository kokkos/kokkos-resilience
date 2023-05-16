#include "Registration.hpp"
#include "resilience/view_hooks/ViewHolder.hpp"
#include "resilience/context/ContextBase.hpp"

namespace KokkosResilience::Detail {
struct ViewHolderRegistration : public RegistrationBase {
  ViewHolderRegistration() = delete;

  ViewHolderRegistration(ContextBase& ctx, const KokkosResilience::ViewHolder& view) : 
    RegistrationBase(view->label()), m_view(view), m_ctx(ctx) {};

  const serializer_t serializer() const override{
    return [&, this](std::ostream& stream){
      size_t buffer_size = need_buffer ? m_view->data_type_size()*m_view->span() : 0;
      char* buf = m_ctx.get_buffer(buffer_size);
  
      m_view->serialize(stream, buf);
      return stream.good();
    };
  }

  const deserializer_t deserializer() const override{
    return [&, this](std::istream& stream){
      size_t buffer_size = need_buffer ? m_view->data_type_size()*m_view->span() : 0;
      char* buf = m_ctx.get_buffer(buffer_size);
  
      m_view->deserialize(stream, buf);
      return stream.good();
    };
  }

  const bool is_same_reference(const Registration& other_reg) const override{
    auto other = dynamic_cast<ViewHolderRegistration*>(other_reg.get());
    
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

  const bool need_buffer = 
  #ifdef KR_ENABLE_MAGISTRATE
      false;
  #else
      !(m_view->span_is_contiguous() && m_view->is_host_space());
  #endif

  ContextBase& m_ctx;
};
}

namespace KokkosResilience {
  template<typename Traits> //Unused
  struct create_registration<KokkosResilience::ViewHolder, Traits>{
    using RegT = Detail::ViewHolderRegistration;
    std::shared_ptr<RegT> reg;

    create_registration(ContextBase& ctx, const KokkosResilience::ViewHolder& view, std::string unused = "") 
        : reg(std::make_shared<RegT>(ctx, view)) {};

    auto get() {
      return std::move(reg);
    }
  };
}
