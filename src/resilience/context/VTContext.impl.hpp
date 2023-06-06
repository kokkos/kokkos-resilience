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

#include "VTContext.hpp"
#include "resilience/util/VTUtil.hpp"

namespace KokkosResilience::Detail {
  template<typename ProxyT>
  VTProxyHolder::VTProxyHolder(ProxyT proxy)
      : KokkosResilience::Util::VT::ProxyID(proxy) {
    using namespace KokkosResilience::Util::VT;

    label = proxy_label(proxy, *this);

    send = [proxy](VTActionMsg msg){
      using ObjT = typename elm_type<ProxyT>::type;
      if constexpr(is_col<ProxyT>::value) {
        if constexpr(is_elm<ProxyT>::value) {
          proxy.template send< &VTContext::template col_action_handler<ObjT> >(msg);
        } else {
          proxy.template broadcast< &VTContext::template col_action_handler<ObjT> >(msg);
        }
      } else {
        if constexpr(is_elm<ProxyT>::value) {
          proxy.template send< &VTContext::template obj_action_handler<ObjT> >(msg);
        } else {
          proxy.template broadcast< &VTContext::template obj_action_handler<ObjT> >(msg);
        }
      }
    };

    is_local = [proxy]() {
      //groups are local to every rank.
      bool local = true;
      if constexpr(is_elm<ProxyT>::value) {
        if constexpr(is_col<ProxyT>::value) {
          local = proxy.tryGetLocalPtr() != nullptr;
        } else {
          local = proxy.getNode() == vt::theContext()->getNode();
        }
      }
      return local;
    };

    register_other = [proxy](VTContext* ctx, 
                            uint64_t other_index_bits){
      if constexpr(not is_elm<ProxyT>::value){
        ctx->register_element(proxy, other_index_bits);
      } else if constexpr(is_obj<ProxyT>::value) {
        ctx->register_element(vt::theObjGroup()->proxyGroup(proxy), other_index_bits);
      } else {
        using ObjT = typename elm_type<ProxyT>::type;
        ctx->register_element(VTCol<ObjT>(proxy.getCollectionProxy()), other_index_bits);
      }
    };

    registration = custom_registration(
        [&, proxy](std::ostream& stream){
          checkpoint::serializeToStream<CheckpointDeps>(*this, stream);
          if constexpr (is_elm<ProxyT>::value) {
            checkpoint::serializeToStream<vt::vrt::CheckpointTrait>(proxy, stream);
          }
          return bool(stream);
        },
        [&, proxy](std::istream& stream){
          checkpoint::deserializeInPlaceFromStream<CheckpointDeps>(stream, this);
          if constexpr (is_elm<ProxyT>::value) {
            checkpoint::deserializeInPlaceFromStream<vt::vrt::CheckpointTrait>(stream, &proxy);
          }
          return bool(stream);
        },
        label + "_impl"
    );
  }

  template<typename SerializerT>
  void VTProxyHolder::serialize(SerializerT& s) {
    s | proxy_bits | index_bits | label | checkpointed_version;
    
    
    //Don't write dependencies to default save file, so we can
    //checkpoint non-proxy user data immediately, instead of waiting
    //for dependency info to come back.
    if(s.hasTraits(CheckpointDeps())){
      if(s.isUnpacking()) dependencies.clear();
      s | dependencies;
    }
  }
}

namespace KokkosResilience {
  
  template<typename T>
  Detail::VTProxyHolder& VTContext::get_holder(T proxy, bool should_mark_modified){
    auto iter = proxy_registry.find(proxy);
    if(iter == proxy_registry.end()){
      //Some bookkeeping for new registrations.
      iter = proxy_registry.emplace(proxy, proxy).first;

      auto& holder = iter->second;

      //Keep a map from group to some registered entry to it,
      //for recovery
      if(holder.is_element()){
        ProxyID group_id = holder;
        group_id.index_bits = -1;
        groups.emplace(group_id, holder);
      } else {
        groups.emplace(holder, holder);
      }
      
      m_backend->register_member(holder.registration.value());

      //Groups should be globally registered and have a one-sided dependence on
      //their elements.
      if constexpr (not Util::VT::is_elm<T>::value){
        msg_before_checkpoint<&VTContext::register_group<T> >(contexts_proxy, proxy, should_mark_modified);

        //Each node marks its local elements as dependencies.
        if constexpr(Util::VT::is_obj<T>::value){
          auto& elm_holder = get_holder(proxy[vt::theContext()->getNode(), should_mark_modified]);
          holder.dependencies[elm_holder] = elm_holder.checkpointed_version;
        } else {
          for(auto index : vt::theCollection()->getLocalIndices(proxy)){
            auto& elm_holder = get_holder(proxy[index], should_mark_modified);
            holder.dependencies[elm_holder] = elm_holder.checkpointed_version;
          }
        }
      }
    }

    auto& holder = iter->second;
    if(should_mark_modified) {
      mark_modified(proxy, holder);
    }

    return holder;
  }

  template<typename GroupProxyT>
  void VTContext::register_group(GroupProxyT group, bool should_mark_modified) { 
    get_holder(group, should_mark_modified);
  }

  template<typename GroupProxyT>
  void VTContext::register_element(GroupProxyT group, uint64_t index_bits) {
    using namespace Util::VT;

    const bool NO_MARK_MODIFIED = false;
    if(index_bits == uint64_t(-1)){
      get_holder(group, NO_MARK_MODIFIED);
    }

    if constexpr(is_obj<GroupProxyT>::value){
      get_holder(group[index_bits], NO_MARK_MODIFIED);
    } else {
      using IndexT = typename elm_type<GroupProxyT>::type::IndexType;
      IndexT index = IndexT::uniqueBitsToIndex(index_bits);
      get_holder(group[index], NO_MARK_MODIFIED);
    }
  }

  template<auto func, typename... Args>
  void VTContext::msg_before_checkpoint(VTContextElmProxy dest, Args... args){
    if(active_region){
      dest.template send<func>(args...);
    } else {
      vt::runInEpochRooted([&]{
        dest.template send<func>(args...);
      });
    }
  }

  template<auto func, typename... Args>
  void VTContext::msg_before_checkpoint(VTContextProxy dest, Args... args){
    if(active_region){
      dest.template broadcast<func>(args...);
    } else {
      vt::runInEpochRooted([&]{
        dest.template broadcast<func>(args...);
      });
    }
  }

  
  template<typename ProxyT>
  void VTContext::mark_modified(ProxyT& proxy,
                                Detail::VTProxyHolder& holder,
                                bool is_remote_request){
    //Remote requests already checked
    bool already_marked = !is_remote_request && 
        modified_proxies.find(proxy) != modified_proxies.end();
    if(already_marked) return;
    
    //Note: Groups are local everywhere, only elements can be non-local.
    if(holder.is_local()){
      modified_proxies.insert(proxy);
    } else {
      msg_before_checkpoint(holder, VTAction::MARK_MODIFIED);
    }

    //Groups should be marked at every rank, but if this
    //was requested by a remote rank, they'll handle it.
    if(!holder.is_element() && !is_remote_request){
      msg_before_checkpoint<&VTContext::remotely_modified<ProxyT>>(contexts_proxy, proxy);

      //Also broadcast to each element to mark modified at their
      //local context
      msg_before_checkpoint(holder, VTAction::MARK_MODIFIED);
    }
  }

  template<typename ProxyT>
  void VTContext::remotely_modified(ProxyT proxy){
    bool already_marked = 
      modified_proxies.find(proxy) != modified_proxies.end();
    if(already_marked) return;

    //Don't mark modified in get_holder
    auto& holder = get_holder(proxy, false);

    //Mark modified as a remote request
    mark_modified(proxy, holder, true);
  }

  //static
  template<typename ObjT>
  void VTContext::col_action_handler(ObjT* obj, VTActionMsg msg){
    auto group_bits = vt::theCollection()->template 
        queryProxyContext<typename ObjT::IndexType>();
    auto group_proxy = Util::VT::VTCol<ObjT>(group_bits);

    auto elm_index = vt::theCollection()->template
        queryIndexContext<typename ObjT::IndexType>();
    auto elm_proxy = group_proxy[*elm_index];

    auto ctx_proxy = vt::theObjGroup()->proxyGroup(msg.sender);
    VTContext* ctx = vt::theObjGroup()->get(ctx_proxy);

    ctx->action_handler(elm_proxy, group_proxy, msg);
  }

  //static
  template<typename ObjT>
  void VTContext::obj_action_handler(ObjT* obj, VTActionMsg msg){
    auto group_proxy = vt::theObjGroup()->getProxy(obj);
    auto elm_proxy = group_proxy[vt::theContext()->getNode()];
    
    auto ctx_proxy = vt::theObjGroup()->proxyGroup(msg.sender);
    VTContext* ctx = vt::theObjGroup()->get(ctx_proxy);

    ctx->action_handler(elm_proxy, group_proxy, msg);
  }

  template<typename ProxyT, typename GroupProxyT>
  void VTContext::action_handler(ProxyT elm,
                                 GroupProxyT group,
                                 const VTActionMsg& msg){
    constexpr bool NO_MARK_MODIFIED = false;
    auto& group_holder = get_holder(group, NO_MARK_MODIFIED);
    auto& elm_holder = get_holder(elm, NO_MARK_MODIFIED);
   
    switch(msg.action){
      case VTAction::REGISTER_PARENT_DEPENDENCY:
        group_holder.dependencies[elm_holder] = elm_holder.checkpointed_version;
        mark_modified(elm, elm_holder);
        break;

      case VTAction::REPORT_VERSION:
        //Wait until we're certain any updates have been made.
        vt::theSched()->runSchedulerWhile([&, requested_epoch = msg.arg]{
          return requested_epoch > prepared_epoch;
        });
        msg.sender.send<&VTContext::set_checkpointed_version<ProxyT>>(
            elm, elm_holder.checkpointed_version
        );
        break;

      case VTAction::MARK_MODIFIED:
        //is_remote_request = true
        mark_modified(elm, elm_holder, true);
        break;

      case VTAction::RESTART:
        restart_proxy(elm_holder, msg.arg);
        break;
    }
  }

  template<typename ProxyT>
  void VTContext::set_checkpointed_version(ProxyT proxy, int version){
    static constexpr bool NO_MARK_MODIFIED = false;
    auto& holder = get_holder(proxy, NO_MARK_MODIFIED);
    holder.checkpointed_version = version;
  }
}


