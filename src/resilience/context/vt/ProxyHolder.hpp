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

#ifndef INC_KOKKOS_RESILIENCE_CONTEXT_VT_PROXYHOLDER_HPP
#define INC_KOKKOS_RESILIENCE_CONTEXT_VT_PROXYHOLDER_HPP

#include <any>
#include <unordered_map>
#include <string>
#include <functional> //std::function

#include <checkpoint/checkpoint.h>
#include "./common.hpp"
#include "resilience/registration/Registration.hpp"

namespace KokkosResilience::Context::VT {

//Magistrate trait for serializing ProxyStatus dependency info or not
struct BasicCheckpointTrait {};

struct ProxyStatus {
  bool tracked = true;
  bool deleted = false;

  int checkpointed_version = -1;
  int restarted_version = -1;

  //dependency ID -> minimum version
  std::unordered_map<ProxyID, int> deps;

  template<typename SerT>
  void serialize(SerT& s) {
    s | tracked | deleted | checkpointed_version;

    if(!s.hasTraits(BasicCheckpointTrait()) && tracked && !deleted) {
      if(s.isUnpacking()) deps.clear();
      s | deps;
    }
  }
};


/**
 *  Type-erasure tool that also holds checkpoint/restart status info.
 *
 *  Note that status info is only maintained at whichever node the proxy
 *  currently resides at, but the holder is valid anywhere. This means prior 
 *  to using any status info, you must either confirm that the proxy is local
 *  or fetch the status and wait on the returned epoch.
 *
 *  Functions denoted local are always local, else any given function may
 *  involve message passing.
 *
 */

class ProxyHolder : public ProxyID {
public:
  //Not to be used, for Magistrate cooperation only.
  ProxyHolder();

  template<typename ProxyT>
  ProxyHolder(ProxyT proxy, VTContext& context); 

  //Should not be copied
  ProxyHolder(ProxyHolder&) = delete;
 
  ~ProxyHolder();

  //Local.
  const std::string& label() const;

  //Local. Collections/ObjGroups are never local.
  bool is_local();
  //Local. Check if element belongs to a dynamic Collection.
  bool is_dynamic();

  //Local. Construct holder to another member of my group
  ProxyHolder* get_holder(ProxyID index_id);
  


  //Fetches updated status, aggregates multiple calls during the window
  //of the first. Does not update dependency information.
  vt::EpochType fetch_status(const std::string& epoch_label);
  
  //Should/shouldn't checkpoint data
  void tracked(bool);
  bool tracked(); //Local.

  //Should delete if present during recovery
  //Deleted proxies are untracked, but restore to
  //previous state on undeleted
  void deleted(bool);
  bool deleted(); //Local.

  void checkpointed_version(int version);
  int checkpointed_version(); //Local.

  void restarted_version(int version);
  int restarted_version(); //Local.

  std::unordered_map<ProxyID, int>& deps(); //Local.


  
  //This proxy will be checkpointed alongside (but separately from)
  //the next checkpoint of a region. Implies deleted(false)
  void modified();

  //Notify that this proxy was (de)registered by user.
  //Implies modified, but may update tracked/untracked state first.
  void registered(std::string region_label);
  void deregistered(std::string region_label);

  
  //Handle proxy having been migrated.
  void migrated_out();
  void migrated_in();

  //Local. Set new status after migrating in. Some function calls on this 
  //will be blocked between migrated_in and migrated_status calls
  void migrated_status(const ProxyStatus& new_status);

  
  //Asynchronously check if this collection element exists, 
  //and if not add to the set
  void check_missing(std::unordered_set<ProxyID>* missing);


  //Registration for the actual proxy data, obeying tracked/untracked/deleted state
  Registration data_registration();
  //Registration for just the metadata
  Registration metadata_registration();

  template<typename SerializerT>
  void serialize(SerializerT& s);
  
protected:
  friend VTContext;

  //Invoke a ProxyAction on typed proxy this holds.
  //Generally, the member functions above should be 
  //prefered to directly invoking - std::any_cast 
  //is inflexible regarding type conversion.
  template<typename ArgT>
  std::any invoke(ProxyAction action, ArgT&& arg){
    return invoker(action, arg); 
  };
  std::any invoke(ProxyAction action);

  //Managed by VTContext, not here.
  vt::EpochType restart_epoch = vt::no_epoch;

  //Both local and do not include/change dependency info
  //  (since deps are only used by the local node)
  ProxyStatus get_status();
  void set_status(const ProxyStatus& new_status);
  
  ProxyStatus _status;
private:
  //If element was migrated in and is still being initialized, wait
  void wait_if_migrating();

  //Registration which handles serializing according to state.
  Registration data_reg;
  
  template<typename ProxyT>
  Registration build_registration(ProxyT proxy);
  
  
  //Type-erasure lambda.
  std::function<std::any(ProxyAction, std::any)> invoker;

  VTContext* ctx;
  
  vt::EpochType fetch_epoch = vt::no_epoch;
  vt::EpochType migrate_in_epoch = vt::no_epoch;
  bool got_status_before_migrate = false;

  //ID for collection event listener registration.
  int listener_id = -1;
};

template<typename ProxyT>
Registration ProxyHolder::build_registration(ProxyT proxy) {
  return custom_registration(
    [this, proxy](std::ostream& stream) {
      if(is_element() && tracked()){
        checkpoint::serializeToStream<vt::vrt::CheckpointTrait>(proxy, stream);
      }
      return bool(stream);
    },
    [this, proxy](std::istream& stream) {
      if(is_element() && tracked()){
        checkpoint::deserializeInPlaceFromStream<vt::vrt::CheckpointTrait, std::istream, decltype(proxy)>(stream, &proxy);
      }
      return bool(stream);
    },
    proxy_label(proxy) + "_impl"
  );
}

}

#endif
