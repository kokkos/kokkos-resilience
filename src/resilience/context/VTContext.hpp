/*
 *
 *                        Kokkos v. 3.0
 *       Copyright (2020) National Technology & Engineering
 *               Solutions of Sandia, LLC (NTESS).
 *
 * Under the terms of Contract DE-NA0003525 with NTESS,
 * the U.S. Government retains certain rights in this software.
 *
 * Kokkos is licensed under 3-clause BSD terms of use:
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 * 1. Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the Corporation nor the names of the
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Questions? Contact Christian R. Trott (crtrott@sandia.gov)
 */
#ifndef INC_KOKKOS_RESILIENCE_VTCONTEXT_HPP
#define INC_KOKKOS_RESILIENCE_VTCONTEXT_HPP

#include <stdexcept>
#include <optional>
#include <type_traits>
#include <vt/vt.h>
#include <checkpoint/checkpoint.h>
#include "ContextBase.hpp"

namespace KokkosResilience {
  class VTContext;

namespace Detail {
  using VTProxyType = std::tuple<uint64_t, uint64_t>;

namespace VTTemplates {
  using VTProxyType = Detail::VTProxyType;
  
  template<typename T>
  using VTCol = vt::vrt::collection::CollectionProxy<T, typename T::IndexType>;
  template<typename T>
  using VTColElm = vt::vrt::collection::VrtElmProxy<T, typename T::IndexType>;
  template<typename T>
  using VTObj = vt::objgroup::proxy::Proxy<T>;
  template<typename T>
  using VTObjElm = vt::objgroup::proxy::ProxyElm<T>;

  template<typename T, typename as = void>
  struct is_col { static constexpr bool value = false; };
  template<typename T, typename as = void>
  struct is_col_elm { static constexpr bool value = false; };
  template<typename T, typename as = void>
  struct is_obj { static constexpr bool value = false; };
  template<typename T, typename as = void>
  struct is_obj_elm { static constexpr bool value = false; };

  template<typename T, typename as = void, typename enable = void*>
  struct is_proxy { static constexpr bool value = false; };

  template<typename T, typename as = void, typename enable = void*>
  struct is_elm { static constexpr bool value = false; };


  template<typename T, typename as>
  struct is_col<VTCol<T>, as> {
    using type = as;
    static constexpr bool value = true;
  };
  template<typename T, typename as>
  struct is_col_elm<VTColElm<T>, as> {
    using type = as;
    static constexpr bool value = true;
  };
  template<typename T, typename as>
  struct is_obj<VTObj<T>, as> {
    using type = as;
    static constexpr bool value = true;
  };
  template<typename T, typename as>
  struct is_obj_elm<VTObjElm<T>, as> {
    using type = as;
    static constexpr bool value = true;
  };

  template<typename T, typename as>
  struct is_elm<T, as, std::enable_if_t<is_col_elm<T>::value or is_obj_elm<T>::value>*> {
    using type = as;
    static constexpr bool value = true;
  };

  template<typename T, typename as>
  struct is_proxy<T, as,
    std::enable_if_t<is_col<T>::value or is_col_elm<T>::value 
                  or is_obj<T>::value or is_obj_elm<T>::value>*
  > {
    using type = as;
    static constexpr bool value = true;
  };


  template<typename T>
  std::string get_label(VTCol<T> proxy){
    return vt::theCollection()->getLabel(proxy.getProxy());
  }
  template<typename T>
  std::string get_label(VTColElm<T> proxy){
    return vt::theCollection()->getLabel(proxy.getCollectionProxy()) 
           + proxy.getIndex().toString();
  }
  template<typename T>
  std::string get_label(VTObj<T> proxy){
    return vt::theObjGroup()->getLabel(proxy);
  }
  template<typename T>
  std::string get_label(VTObjElm<T> proxy){
    return vt::theObjGroup()->getLabel(vt::theObjGroup()->proxyGroup(proxy))
           + std::to_string(proxy.getNode());
  }
  
  template<typename T>
  VTProxyType get_proxy(VTCol<T> proxy) { return {proxy.getProxy(), -1}; }
  template<typename T>
  VTProxyType get_proxy(VTColElm<T> proxy) { return {proxy.getCollectionProxy(), proxy.getElementProxy().getIndex().uniqueBits()}; }
  template<typename T>
  VTProxyType get_proxy(VTObj<T> proxy) { return {proxy.getProxy(), -1}; }
  template<typename T>
  VTProxyType get_proxy(VTObjElm<T> proxy) { return {proxy.getProxy(), proxy.getNode()}; }
  
  enum GenericAction {
    CHECKPOINT,
    RESTORE,
    REPORT_VERSION,
    REGISTER_PARENT_DEPENDENCY,
    MARK_MODIFIED
  };
  struct GenericActionMsg {
    VTObjElm<VTContext> sender;
    GenericAction action;
    VTProxyType arg1;
    int arg2;
  };

  template<typename T>
  void ColActionHandler(T* obj, GenericActionMsg msg);
  template<typename T>
  void ObjActionHandler(T* obj, GenericActionMsg msg);

  using GenericActionSender = std::function<void (GenericActionMsg)>;

  template<typename T>
  GenericActionSender
  make_sender(VTCol<T> proxy){
    return [proxy](GenericActionMsg msg){
      proxy.template broadcast<ColActionHandler<T>>(msg);
    };
  }
  template<typename T>
  GenericActionSender
  make_sender(VTColElm<T> proxy){
    return [proxy](GenericActionMsg msg){
      proxy.template send<ColActionHandler<T>>(msg);
    };
  }
  template<typename T>
  GenericActionSender
  make_sender(VTObj<T> proxy){
    return [proxy](GenericActionMsg msg){
      proxy.template broadcast<ObjActionHandler<T>>(msg);
    };
  }
  template<typename T>
  GenericActionSender
  make_sender(VTObjElm<T> proxy){
    return [proxy](GenericActionMsg msg){
      proxy.template send<ObjActionHandler<T>>(msg);
    };
  }

    
  struct VTProxyStatus {

    VTProxyStatus() {
      //Magistrate needs a default constructor, but
      //we only want to allow deserializeInPlace
      assert(false && "Default constructor should not be used unless performing deserializeInPlace!");
    };

    template<typename T>
    VTProxyStatus(T proxy)
      : is_element(is_elm<T>::value),
        send(make_sender(proxy)),
        label(get_label(proxy)) {

      is_local = [proxy]{
        bool local;
        if constexpr (not is_elm<T>::value){
          local = false;
        } else if constexpr (is_obj_elm<T>::value){
          local = vt::theContext()->getNode() == proxy.getNode();
        } else {
          local = proxy.tryGetLocalPtr() != nullptr;
        }
        return local;
      };

      if (!is_element) return;

      registration = custom_registration(
          [&, proxy](std::ostream& stream){
            checkpoint::serializeToStream(*this, stream);
            checkpoint::serializeToStream<vt::vrt::CheckpointTrait>(proxy, stream);
            return bool(stream);
          },
          [&, proxy](std::istream& stream){
            checkpoint::deserializeInPlaceFromStream(stream, this);
            checkpoint::deserializeInPlaceFromStream<vt::vrt::CheckpointTrait>(stream, &proxy);
            return bool(stream);
          },
          label + "_impl"
      );
    };
    
    //last checkpointed id (iteration, distributed timer, etc)
    int checkpointed_version = -1;
    //Last time we asked the owner of this object which version it has.
    int queried_version = -1;
  
    //required versions of dependencies.
    std::unordered_map<VTProxyType, size_t> dependencies;

    const bool is_element = false;
    const GenericActionSender send = nullptr;
    const std::string label = "";

    using LocalChecker = std::function<bool ()>;
    LocalChecker is_local = []{return true;};

    std::optional<Registration> registration;

    void operator()(VTObjElm<VTContext> sender, GenericAction action, VTProxyType arg = {0,0}){
      GenericActionMsg msg;
      msg.sender = sender;
      msg.action = action;
      msg.arg1 = arg;
      send(msg);
    }

    void operator()(VTObjElm<VTContext> sender, GenericAction action, int arg){
      GenericActionMsg msg;
      msg.sender = sender;
      msg.action = action;
      msg.arg2 = arg;
      send(msg);
    }

    template<typename SerT>
    void serialize(SerT& s){
      s | checkpointed_version;
      s | dependencies;
    }
  };
}
}

class VTContext : public ContextBase {
public:
  explicit VTContext(const std::string& config_file)
      : ContextBase(config_file, vt::theContext()->getNode()), 
        context_proxy(vt::theObjGroup()->makeCollective(this, "kr::VTContext")) {}
 
  using VTProxyType      = Detail::VTProxyType;
  using VTProxyStatus    = Detail::VTTemplates::VTProxyStatus;
  using ContextGroupType = Detail::VTTemplates::VTObj<VTContext>;
  using ContextElmType   = Detail::VTTemplates::VTObjElm<VTContext>;
 
  VTContext(const VTContext &)     = delete;
  VTContext(VTContext &&) noexcept = default;
 
  VTContext &operator=(const VTContext &) = delete;
  VTContext &operator=(VTContext &&) noexcept = default;
 
  virtual ~VTContext() {
    vt::theObjGroup()->destroyCollective(context_proxy);
  }
 
  bool restart_available(const std::string &label, int version) override {
    return m_backend->restart_available(label, version);
  }
 
  void restart(const std::string &label, int version,
               const std::unordered_set<Registration> &members) override {
    m_backend->restart(label, version, members);
  }
 
  void checkpoint(const std::string &label, int version,
                  const std::unordered_set<Registration> &members) override {
    checkpoint_proxies();
 
    m_backend->checkpoint(label, version, members);
  }
 
  int latest_version(const std::string &label) const noexcept override {
    return m_backend->latest_version(label);
  }
  
  void reset() override { m_backend->reset(); }
 
  void checkpoint_proxies();
 
  //Hold on to proxy dependencies, last checkpoint stats
  using ProxyMap = std::unordered_map<VTProxyType, VTProxyStatus>;
  ProxyMap proxy_registry;
 
  //Proxies known to have been changed since last checkpoint
  std::unordered_set<VTProxyType> modified_proxies;
 
  //Get the proxy_status of a proxy, making it if need be.
  template<typename T>
  VTProxyStatus& proxy_status(T proxy, bool mark_modified = true){
    using namespace Detail::VTTemplates;
    static_assert(is_proxy<T>::value, "Attempting to register a non-proxy type as a VT proxy");
 
    VTProxyType proxy_bits = get_proxy(proxy);
 
    auto iter = proxy_registry.find(proxy_bits);
 
    if(iter == proxy_registry.end()){
      iter = proxy_registry.emplace(proxy_bits, proxy).first;
      auto& status = iter->second;

      //Elements register to the backend for checkpointing,
      //non-elements register all of their elements.
      if(!status.is_element){
        msg_before_checkpoint(iter->second, REGISTER_PARENT_DEPENDENCY);
      } else {
        m_backend->register_member(iter->second.registration.value());
      }
    }
 
    auto& status = iter->second;
    if(mark_modified){
      //Local elements update locally, non-elements update
      //locally and the other locations.
      if(!status.is_element || status.is_local())
        modified_proxies.insert(proxy_bits);
      if(!status.is_element || !status.is_local())
        msg_before_checkpoint(status, MARK_MODIFIED);
    }
 
    return status;
  }

  template<typename... Args>
  void msg_before_checkpoint(VTProxyStatus& status, Args... args){
    if(active_region){
      //If we've in an active region, we know
      //messages here will finish before the
      //checkpoint is called.
      status(m_proxy, args...);
    } else {
      vt::runInEpochRooted([&]{
        status(m_proxy, args...);
      });
    }
  }
 
  void update_dependency(VTProxyType dep, int version){
    proxy_registry[dep].checkpointed_version = version;
  }
 
  void checkpoint_elm(VTProxyType proxy, int version){
    auto& registration = proxy_registry[proxy].registration.value();
    m_backend->checkpoint(registration->name, version, {registration});
  }
 
  void restart_elm(VTProxyType proxy, int version){
    //TODO: We may not have a registration yet on recovery, we
    //can try to find the element by proxy+index, or failing that
    //either delay recovery of this one, or fail?
    auto& registration = proxy_registry[proxy].registration.value();
    m_backend->checkpoint(registration->name, version, {registration});
  }
 
private:
  ContextGroupType context_proxy;
  ContextElmType m_proxy = context_proxy[m_pid];
};


namespace Detail::VTTemplates {
  template<typename Proxy, typename ParentProxy>
  void handle_action(Proxy elm, ParentProxy parent, const GenericActionMsg& msg){
    auto ctx_proxy = vt::theObjGroup()->proxyGroup(msg.sender);
    VTContext* ctx = vt::theObjGroup()->get(ctx_proxy);
   
    constexpr bool NO_MARK_MODIFIED = false;
    VTProxyStatus& status = ctx->proxy_status(elm, NO_MARK_MODIFIED);
    VTProxyStatus& parent_status = ctx->proxy_status(parent, NO_MARK_MODIFIED);

    switch(msg.action){
      case REGISTER_PARENT_DEPENDENCY:
        parent_status.dependencies[get_proxy(elm)] = status.checkpointed_version;
        ctx->modified_proxies.insert(get_proxy(elm));
        ctx->modified_proxies.insert(get_proxy(parent));
        break;
      case REPORT_VERSION:
        msg.sender.send<&VTContext::update_dependency>(get_proxy(elm), status.checkpointed_version);
        break;
      case CHECKPOINT:
        ctx->checkpoint_elm(get_proxy(elm), msg.arg2);
        break;
      case MARK_MODIFIED:
        ctx->modified_proxies.insert(get_proxy(elm));
        break;
      default:
        fprintf(stderr, "Unexpected action type %d!\n", msg.action);
    }
  }

  template<typename T>
  void ColActionHandler(T* obj, GenericActionMsg msg){
    auto col_proxy_bits = vt::theCollection()->template queryProxyContext<typename T::IndexType>();
    auto col_proxy = VTCol<T>(col_proxy_bits);

    auto elm_index = vt::theCollection()->template queryIndexContext<typename T::IndexType>();
    auto elm_proxy = col_proxy[*elm_index];

    handle_action(elm_proxy, col_proxy, msg);
  }

  template<typename T>
  void ObjActionHandler(T* obj, GenericActionMsg msg){
    auto obj_proxy = vt::theObjGroup()->getProxy(obj);
    auto elm_proxy = obj_proxy[vt::theContext()->getNode()];

    handle_action(elm_proxy, obj_proxy, msg);
  }


}

} // namespace KokkosResilience

#endif // INC_KOKKOS_RESILIENCE_VTCONTEXT_HPP
