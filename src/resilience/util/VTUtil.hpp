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

#ifndef INC_KOKKOS_RESILIENCE_UTIL_VT_HPP
#define INC_KOKKOS_RESILIENCE_UTIL_VT_HPP

#include <type_traits>
#include <tuple>
#include <vt/vt.h>

namespace KokkosResilience::Util::VT {
  template<typename T>
  using VTCol = vt::vrt::collection::CollectionProxy<T, typename T::IndexType>;
  template<typename T>
  using VTColElm = vt::vrt::collection::VrtElmProxy<T, typename T::IndexType>;
  template<typename T>
  using VTObj = vt::objgroup::proxy::Proxy<T>;
  template<typename T>
  using VTObjElm = vt::objgroup::proxy::ProxyElm<T>;

  //Get the type of the actual elements referenced by the proxy.
  template<typename T>
  struct _elm_type;

  template<typename T>
  struct _elm_type<VTCol<T>> {
    using type = T;
  };
  template<typename T>
  struct _elm_type<VTColElm<T>>{
    using type = T;
  };
  template<typename T>
  struct _elm_type<VTObj<T>>{
    using type = T;
  };
  template<typename T>
  struct _elm_type<VTObjElm<T>>{
    using type = T;
  };

  template<typename T>
  struct elm_type : public _elm_type<typename std::decay<T>::type> {};

  
  //Any collection or its elements
  template<typename T, typename as>
  struct _is_col { static constexpr bool value = false; };

  template<typename T, typename as>
  struct _is_col<VTCol<T>, as> {
    using type = as;
    static constexpr bool value = true;
  };

  template<typename T, typename as>
  struct _is_col<VTColElm<T>, as> : public _is_col<VTCol<T>, as> {};
  
  template<typename T, typename as = void>
  struct is_col : public _is_col<typename std::decay<T>::type, as> {};
  
  
  //Any objgroup or its elements
  template<typename T, typename as>
  struct _is_obj { static constexpr bool value = false; };
  
  template<typename T, typename as>
  struct _is_obj<VTObj<T>, as> {
    using type = as;
    static constexpr bool value = true;
  };
  
  template<typename T, typename as>
  struct _is_obj<VTObjElm<T>, as> : public _is_obj<VTObj<T>, as> {};
  
  template<typename T, typename as = void>
  struct is_obj : public _is_obj<typename std::decay<T>::type, as> {};
  
  
  //Element of any objgroup/collection
  template<typename T, typename as>
  struct _is_elm { static constexpr bool value = false; };
  
  template<typename T, typename as>
  struct _is_elm<VTColElm<T>, as> {
    using type = as;
    static constexpr bool value = true;
  };
  
  template<typename T, typename as>
  struct _is_elm<VTObjElm<T>, as> {
    using type = as;
    static constexpr bool value = true;
  };

  template<typename T, typename as = void>
  struct is_elm : public _is_elm<typename std::decay<T>::type, as> {};
  
  
  //Any objgroup/collection and their elements
  template<typename T, typename as = void, typename enable = void*>
  struct is_proxy { static constexpr bool value = false; };

  template<typename T, typename as>
  struct is_proxy<T, as,
    typename std::enable_if_t<is_col<T>::value or is_obj<T>::value, void*>
  > {
    using type = as;
    static constexpr bool value = true;
  };


  struct ProxyID {
    template<
      typename ProxyT,
      typename enable = typename is_proxy<ProxyT, void*>::type
    >
    ProxyID(ProxyT&& proxy) : 
      proxy_bits(get_proxy_bits(std::forward<ProxyT>(proxy))),
      index_bits(get_index_bits(std::forward<ProxyT>(proxy))) { };

    ProxyID() = default;

    bool operator==(const ProxyID& other) const {
      return proxy_bits == other.proxy_bits && index_bits == other.index_bits;
    }

    bool is_element() const {
      return index_bits != uint64_t(-1);
    }

    template<typename SerializerT>
    void serialize(SerializerT& s){
      s | proxy_bits | index_bits;
    }

    uint64_t proxy_bits;
    uint64_t index_bits;

    //The indexless collection/objgroup from
    //an element within it.
    ProxyID get_group_id(){
      ProxyID group = *this;
      group.index_bits = -1;
      return group;
    }

  private:
    template<typename ProxyT>
    uint64_t get_proxy_bits(ProxyT&& proxy){
      if constexpr(is_col<ProxyT>::value and is_elm<ProxyT>::value){
        return proxy.getCollectionProxy();
      } else {
        return proxy.getProxy();
      }
    }

    template<typename ProxyT>
    uint64_t get_index_bits(ProxyT&& proxy){
      if constexpr(not is_elm<ProxyT>::value){
        return -1;
      } else if constexpr(is_col<ProxyT>::value){
        return proxy.getElementProxy().getIndex().uniqueBits();
      } else {
        return proxy.getNode();
      }
    }
  };
}

namespace std {
  //Hash as if it were just a tuple.
  template<>
  struct hash<KokkosResilience::Util::VT::ProxyID> {
    size_t operator()(const KokkosResilience::Util::VT::ProxyID& id) const {
      return hash< std::tuple<uint64_t, uint64_t> >()(
          make_tuple(id.proxy_bits, id.index_bits)
      );
    }
  };
} 

namespace KokkosResilience::Util::VT {

  template<typename ProxyT, typename enable = typename is_proxy<ProxyT, void*>::type>
  std::string proxy_label(ProxyT proxy, const ProxyID& id){
    std::string label;

    if constexpr (is_col<ProxyT>::value) {
      label = vt::theCollection()->getLabel(id.proxy_bits);
      if constexpr (is_elm<ProxyT>::value) {
        label += proxy.getIndex().toString();
      }
    } else {
      if constexpr (is_elm<ProxyT>::value) {
        label = vt::theObjGroup()->getLabel(vt::theObjGroup()->proxyGroup(proxy));
        label += "[" + std::to_string(proxy.getNode()) + "]";
      } else {
        label = vt::theObjGroup()->getLabel(proxy);
      }
    }
    return label;
  }

  template<typename ProxyT>
  inline std::string proxy_label(ProxyT proxy){
    return proxy_label(proxy, proxy);
  }

  template<typename ProxyT>
  inline bool is_local(ProxyT&& proxy){
    if constexpr(not is_elm<ProxyT>::value){
      return false;
    } else if constexpr(is_col<ProxyT>::value){
      return proxy.tryGetLocalPtr() != nullptr;
    } else {
      return proxy.getNode() == vt::theContext()->getNode();
    }
  }

  //Send if proxy is an element, else broadcast
  template<auto F, typename ProxyT, typename MsgT>
  inline void msg(ProxyT proxy, MsgT& msg){
    if constexpr(is_elm<ProxyT>::value) {
      proxy.template send<F>(msg);
    } else {
      proxy.template broadcast<F>(msg);
    }
  }

  //Returns a proxy guaranteed to be a Collection/ObjGroup
  template<typename ProxyT>
  auto deindex(ProxyT proxy){
    if constexpr(not is_elm<ProxyT>::value){
      return proxy;
    } else if constexpr(is_col<ProxyT>::value){
      using ObjT = typename elm_type<ProxyT>::type;
      return VTCol<ObjT>(proxy.getCollectionProxy());
    } else {
      return vt::theObjGroup()->proxyGroup(proxy);
    }
  }

  //Returns proxy belonging to same group but with the given index
  template<typename ProxyT>
  auto reindex(ProxyT proxy, uint64_t index) {
    if constexpr(is_col<ProxyT>::value){
      //Collections indexed by custom type, whose storage is type-defined
      using IndexT = typename elm_type<ProxyT>::type::IndexType;
      IndexT typed_index = IndexT::uniqueBitsToIndex(index);
      
      return deindex(proxy)[typed_index];
    } else {
      //ObjGroups indexed by rank, which is directly stored as uint64_t
      return deindex(proxy)[index];
    }
  }

}


#endif
