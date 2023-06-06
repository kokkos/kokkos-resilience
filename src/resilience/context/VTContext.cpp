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

namespace KokkosResilience {
std::unique_ptr< ContextBase >
make_context( vt::ctx::Context* vtTheCtx, const std::string &config )
{
  return std::make_unique<VTContext>(config);
}

void VTContext::begin_operation(){
  //Make sure we're done w/ any previous checkpoint/recovery operations
  if(cur_epoch != vt::no_epoch) vt::runSchedulerThrough(cur_epoch);

  //Make sure the next one had time to be prepared
  if(next_epoch == vt::no_epoch){
    vt::theSched()->runSchedulerWhile([&]{return next_epoch == vt::no_epoch;});
  }

  cur_epoch = next_epoch;
  next_epoch = vt::no_epoch;

  vt::theMsg()->pushEpoch(cur_epoch);
}

void VTContext::end_operation(){
  vt::theMsg()->popEpoch();

  vt::theTerm()->addAction(cur_epoch, [&]{
    //Start building the next epoch once this one is done
    //This way we can continue w/o having to wait on the
    //collective epoch generation in the main thread.
    next_epoch = vt::theTerm()->makeEpochCollective("kr::VTContext checkpoint " + std::to_string(prepared_epoch+1) + " epoch\n");
  });
  
  vt::theTerm()->finishedEpoch(cur_epoch);

  //For now, require exiting as a group.
  vt::runSchedulerThrough(cur_epoch);
}

void VTContext::msg_before_checkpoint(
    VTProxyHolder& holder, VTAction action, int arg) {
  auto msg = VTActionMsg(m_proxy, action, arg);
  if(active_region){
    //If we're in an active region, we know
    //messages here will finish before the
    //checkpoint is called.
    holder.send(msg);
  } else {
    vt::runInEpochRooted([&]{
      holder.send(msg);
    });
  }
}

void VTContext::checkpoint_proxies(){
  //Update checkpointed version beforehand, so dependencies
  //also checkpointing now see this version as the required.
  for(auto& modified_proxy : modified_proxies){
    auto& holder = proxy_registry[modified_proxy];
    holder.checkpointed_version++;
  }

  //With that updated, we are prepared to handle other
  //rank's queries about checkpointed versions.
  prepared_epoch++;

  //Request the latest version of dependencies for 
  //any elements being checkpointed, then checkpoint
  for(auto& modified_proxy : modified_proxies){
    auto& holder = proxy_registry[modified_proxy];
    holder.epoch = vt::theTerm()->makeEpochRooted("kr::VTContext checkpoint " + std::to_string(prepared_epoch) + " gather dependencies for " + holder.label);

    vt::theMsg()->pushEpoch(holder.epoch);
    //We need to query non-local dependencies' most recent checkpoint
    for(auto& dependency : holder.dependencies){
      auto& dep_holder = proxy_registry[dependency.first];

      if(dep_holder.is_local()) continue;

      if(dep_holder.epoch == vt::no_epoch){
        //Put request in it's own epoch, so multiple requesters can independently depend
        dep_holder.epoch = vt::theTerm()->makeEpochRooted("kr::VTContext checkpoint " + std::to_string(prepared_epoch) + " request version of " + dep_holder.label);
        vt::theMsg()->pushEpoch(dep_holder.epoch);

        //Have remote proxy update checkpoint_version here, ensuring we get
        //checkpoint_version for this checkpoint not any older ones
        dep_holder.send(VTActionMsg(m_proxy, VTAction::REPORT_VERSION, prepared_epoch));

        vt::theMsg()->popEpoch();
        vt::theTerm()->finishedEpoch(dep_holder.epoch);

        //Once this checkpoint operation is done, reset
        vt::theTerm()->addAction(cur_epoch, [&]{ dep_holder.epoch = vt::no_epoch; });
      } else {
        //Someone already made the request, just register it as a dependency
        vt::theTerm()->addDependency(holder.epoch, dep_holder.epoch);
      }
    }
    vt::theMsg()->popEpoch();

    //Once queries are returned, we can update deps and checkpoint.
    vt::theTerm()->addAction(holder.epoch, [&]{
      for(auto& dependency : holder.dependencies){
        dependency.second = proxy_registry[dependency.first].checkpointed_version;
      }

      //Individual elements are checkpointed once, globally
      //Collections have their dependency info checkpointed per-rank
      const bool is_global = holder.is_element();

      auto& registration = holder.registration.value();
      m_backend->checkpoint(registration->name,
                            holder.checkpointed_version,
                            {registration}, is_global);
    });
    vt::theTerm()->finishedEpoch(holder.epoch);
  }

  modified_proxies.clear();
}

void VTContext::restart_proxy(Util::VT::ProxyID proxy, int version, bool is_remote_request) {
  using namespace Util::VT;

  auto holder_iter = proxy_registry.find(proxy);

  //Handle not having a registration to this proxy yet
  if(holder_iter == proxy_registry.end()){
    //Try to generate a registration from some other member of the
    //same group
    ProxyID group_id = proxy;
    if(group_id.is_element()) group_id.index_bits = -1;

    auto group_iter = groups.find(group_id);
    if(group_iter == groups.end()){
      fprintf(stderr, "WARNING: could not find or construct registration for a dependency on proxy %lu index %lu\n", proxy.proxy_bits, proxy.index_bits);
      return;
    }

    auto& group_holder = proxy_registry[group_iter->second];
    group_holder.register_other(this, proxy.index_bits);

    holder_iter = proxy_registry.find(proxy);
    assert(holder_iter != proxy_registry.end());
  }


  auto& holder = holder_iter->second;
  
  if(holder.recovered_version != -1){
    assert(holder.recovered_version == version);
    return;
  }

  if(!holder.is_local()){
    holder.send(VTActionMsg(m_proxy, VTAction::RESTART, version));
    holder.recovered_version = version;
    return;
  }

  //Finally, actually restart
  auto& registration = holder.registration.value();
  const bool IS_GLOBAL = holder.is_element();
  m_backend->restart(registration->name, version, {registration}, IS_GLOBAL);

  assert(holder.checkpointed_version == version);
  holder.recovered_version = version;
  
  //Now that we've restarted, we have a list of dependencies to also restart
  //Groups need everyone to be participate, so be sure that's happening.
  if(!is_remote_request){
    contexts_proxy.template broadcast< &VTContext::restart_group >(proxy, version);
  }
  for(auto dep_iter : holder.dependencies) {
    restart_proxy(dep_iter.first, dep_iter.second);
  }
}
  
void VTContext::restart_group(ProxyID group, int version){
  const bool IS_REMOTE_REQUEST = true;
  restart_proxy(group, version, IS_REMOTE_REQUEST);
}

void VTContext::restart_proxies() {
  using namespace Util::VT;

  //Recover anything marked with a checkpointed_version set during
  //  the default member recovery step, and traverse dependencies.
  //For now, fully sychronous recovery. In the future, we may have 
  //  a synchronous dependency traversal for marking what to recover,
  //  then we can asynchronously perform recovery (w/ element locks)
  vt::runInEpochCollective( [&]{
    for(auto& holder_iter : proxy_registry){
      auto& holder = holder_iter.second;
      
      if(holder.checkpointed_version != -1 && holder.recovered_version == -1){
        restart_proxy(holder, holder.checkpointed_version);
      }
    }
  });
}

}
