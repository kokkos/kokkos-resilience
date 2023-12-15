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

#include "VTContext.impl.hpp"

namespace KokkosResilience {

std::unique_ptr< ContextBase >
make_context( vt::ctx::Context* vt_theCtx, const std::string &config )
{
  return std::make_unique<Context::VT::VTContext>(config);
}

}

namespace KokkosResilience::Context::VT {

vt::trace::UserEventIDType VTContext::checkpoint_region = vt::trace::no_user_event_id;
vt::trace::UserEventIDType VTContext::checkpoint_wait = vt::trace::no_user_event_id;
vt::trace::UserEventIDType VTContext::serialize_proxy = vt::trace::no_user_event_id;
vt::trace::UserEventIDType VTContext::offset_region = vt::trace::no_user_event_id;
vt::trace::UserEventIDType VTContext::offset_wait = vt::trace::no_user_event_id;

VTContext::VTContext(const std::string& config_file) 
    : ContextBase(config_file, vt::theContext()->getNode()),
      holders(*this),
      phase_begin_hookid(vt::thePhase()->registerHookRooted(
        vt::phase::PhaseHook::Start,
        [this](){
          in_phase = true;
        }
      )),
      phase_end_hookid(vt::thePhase()->registerHookCollective(
        vt::phase::PhaseHook::EndPostMigration,
        [this](){
          in_phase = false;
          handle_pending_events();
        }
      )),
      max_iteration_offset(get_config_max_iter_offset()),
      contexts_proxy(vt::theObjGroup()->makeCollective(this, "kr::VTContext"))
{ 
  checkpoint_region = vt::trace::registerEventCollective("CheckpointRegion");
  checkpoint_wait = vt::trace::registerEventCollective("CheckpointWait");
  serialize_proxy = vt::trace::registerEventCollective("SerializeProxy");
  offset_region = vt::trace::registerEventCollective("OffsetIterRegion");
  offset_wait = vt::trace::registerEventCollective("OffsetIterWait");
}

VTContext::~VTContext() {
  vt::thePhase()->unregisterHook(phase_begin_hookid);
  vt::thePhase()->unregisterHook(phase_end_hookid);
  vt::theObjGroup()->destroyCollective(contexts_proxy);
}

void VTContext::restart(const std::string& label, int version,
                        const std::unordered_set<Registration>& members) {
  m_backend->restart(label, version, members);
  restart_proxies(label, version);
}

void VTContext::checkpoint(const std::string& label, int version,
                           const std::unordered_set<Registration>& members) {
  if(checkpoint_epoch != vt::no_epoch){
    vt::trace::TraceScopedNote trace_obj(fmt::format("previous checkpoint wait {}@{}", label, version), checkpoint_wait);
    vt::runSchedulerThrough(checkpoint_epoch);
    trace_obj.end();
  }
  checkpoint_epoch = vt::no_epoch;

  //Construct checkpoint_epoch after offset iter finised
  vt::theTerm()->addAction(offset_iter_epoch, [&, parent_epoch = vt::theTerm()->getEpoch()](){
    vt::theTerm()->pushEpoch(parent_epoch);
    checkpoint_proxies(label, version);
    vt::theTerm()->popEpoch(parent_epoch);
  });
  {
  vt::trace::TraceScopedNote trace_obj(fmt::format("offset iter wait {}@{}", label, version), offset_wait);
  vt::runSchedulerThrough(offset_iter_epoch);
  trace_obj.end();
  }
  offset_iter_epoch = vt::no_epoch;

  //Make sure action had a chance to run, then wait for checkpoint_epoch
  if(checkpoint_epoch == vt::no_epoch){
    vt::theSched()->runSchedulerWhile([this](){ return checkpoint_epoch == vt::no_epoch; });
  }
  if(max_iteration_offset == 0 || true){
    vt::trace::TraceScopedNote trace_obj(fmt::format("checkpoint wait {}@{}", label, version), checkpoint_wait);
    vt::runSchedulerThrough(checkpoint_epoch);
    checkpoint_epoch = vt::no_epoch;
    trace_obj.end();
  }
 
  //TODO: Re-enable this when needed for non-proxy members
  //double region_start_s = vt::timing::getCurrentTime();
  //vt::runSchedulerThrough(region_epochs.front());
  //fmt::print("Waiting for region took {}s\n", vt::timing::getCurrentTime() - region_start_s);
  
  m_backend->checkpoint(label, version, members);

  //TODO: This should be handled better when using offset iterations.
  modified_proxies.clear();
}

void VTContext::register_member(Registration member, Region region){
  ProxyHolder* holder = holders[member];
  if(holder) {
    holder->registered(region.label());

    //Don't complete registration if non-local,
    //holder will have sent off for locally registering it.
    if(holder->is_element() && !holder->is_local()) return;
  }
  
  //Now just do default registration.
  ContextBase::register_member(member, region);
}

void VTContext::deregister_member(Registration member, Region region){
  const bool erased = region.members().erase(member);
  const int n_reg = count_registrations(member);
  
  if(erased && n_reg == 0){
    //If last local registration gone, deregister from backend.
    m_backend->deregister_member(member);
  }

  if(n_reg == 0){
    //If last local gone, or no locals here in the first place,
    //mark proxy as deregistered
    ProxyHolder* holder = holders[member];
    if(holder){
      holder->deregistered(region.label());
    }
  }
}

void VTContext::enter_region(Region region, int version){
  if((*active_filter)(version + max_iteration_offset)){
    assert(offset_iter_epoch == vt::no_epoch);

    offset_iter_epoch = vt::theTerm()->makeEpochCollective(
      fmt::format("kr:: {}@{} application work", region.label(), version)
    );

    auto trace_obj = std::make_shared<vt::trace::TraceScopedNote>(
      fmt::format("offset iter region {}@{}", region.label(), version), 
      offset_region
    );
    vt::theTerm()->addAction(offset_iter_epoch, [trace_obj](){
      trace_obj->end();
    });
  
    vt::theMsg()->pushEpoch(offset_iter_epoch);
  }
}

void VTContext::exit_region(Region region, int version){
  if((*active_filter)(version + max_iteration_offset)){
    vt::theMsg()->popEpoch(offset_iter_epoch);
    vt::theTerm()->finishedEpoch(offset_iter_epoch);
  }
}

size_t VTContext::get_config_max_iter_offset(){
  //Default: each region ran in full immediately after exiting
  int max_offset = 0;

  const auto& cfg = this->config()["contexts"]["vt"];
  auto usr_input = cfg.get("max_iteration_offset");

  if(usr_input){
    max_offset = static_cast<size_t>(usr_input->template as<double>());
    if(vt::theContext()->getNode() == 0) fmt::print("kr::VTContext running with max {} iterations offset\n", max_offset);
  }

  return max_offset;
}

void VTContext::handle_pending_events(){
  while(!pending_element_events.empty()){
    auto info = pending_element_events.front();
    pending_element_events.pop_front();
    
    auto holder_ptr = holders[info.first];
    assert(holder_ptr != nullptr);
    auto& holder = *holder_ptr;

    using Event = vt::vrt::collection::listener::ElementEventEnum;
    Event event = info.second;

    handle_element_event(holder, event);
  }
}

void VTContext::handle_element_event(
    ProxyID proxy, 
    vt::vrt::collection::listener::ElementEventEnum event
) {
  auto holder_ptr = holders[proxy];
  assert(holder_ptr != nullptr);
  auto& holder = *holder_ptr;

  //fmt::print(stderr, "{} handling {} for {}.\n", m_proxy, event, holder);

  using Event = vt::vrt::collection::listener::ElementEventEnum;
  switch(event){
    case Event::ElementCreated:
      holders[holder.get_group_id()]->deps()[holder] = -1;
      break;

    case Event::ElementDestroyed:
      holder.deleted();
      holder.modified();
      break;

    case Event::ElementMigratedOut:
      holders[holder.get_group_id()]->deps().erase(holder);
      if(in_phase) pending_element_events.push_back(std::make_pair(proxy, event));
      else holder.migrated_out();
      break;

    case Event::ElementMigratedIn:
      holders[holder.get_group_id()]->deps()[holder] = -1;
      if(in_phase) pending_element_events.push_back(std::make_pair(proxy, event));
      else holder.migrated_in();
      break;
  }
}

void VTContext::checkpoint_proxies(const std::string& region, int version){
  assert(checkpoint_epoch == vt::no_epoch);
  
  checkpoint_epoch = vt::theTerm()->makeEpochCollective(
      fmt::format("{} checkpoint wrapper {}@{}", m_proxy, region, version)
  );
  
  auto trace_obj = std::make_shared<vt::trace::TraceScopedNote>(
    fmt::format("Checkpoint region {}@{}", region, version), 
    checkpoint_region
  );
  vt::theTerm()->addAction(checkpoint_epoch, [trace_obj](){
    trace_obj->end();
  });
  
  vt::theTerm()->pushEpoch(checkpoint_epoch);

  auto epoch_fmt = 
    fmt::format("{} checkpoint {}@{} gather dependencies of {}", m_proxy, region, version, "{}");

  auto fetch_epoch_fmt = 
    fmt::format("{} checkpoint {}@{} fetch status of {}", m_proxy, region, version, "{}");

  //Fetch status of dependencies for any elements being 
  //checkpointed, then checkpoint
  for(auto& modified_proxy : modified_proxies){
    auto& holder = *holders[modified_proxy];
    
    vt::EpochType deps_epoch = vt::theTerm()->makeEpochRooted(
      fmt::format(epoch_fmt, holder)
    );

    for(auto& dep : holder.deps()){
      auto* dep_holder_ptr = holders[dep.first];
      assert(dep_holder_ptr != nullptr);
      auto& dep_holder = *dep_holder_ptr;

      //Only need to fetch non-local dependencies.
      if(dep_holder.is_local()) continue;
      auto fetch_epoch = dep_holder.fetch_status(
        fmt::format(fetch_epoch_fmt, dep_holder)
      );
      vt::theTerm()->addDependency(fetch_epoch, deps_epoch);
    }
    vt::theTerm()->finishedEpoch(deps_epoch);
    

    vt::theTerm()->addLocalDependency(checkpoint_epoch);

    //Once fetches are done, update metadata and checkpoint
    vt::theTerm()->addAction(deps_epoch, [this, &holder](){
      if(max_iteration_offset == 0){
        vt::theSched()->enqueue([this, &holder](){
          checkpoint_proxy(holder, checkpoint_epoch);
        });
      } else {
        vt::theSched()->enqueue([this, &holder](){
        vt::theSched()->getThreadManager()->allocateThreadRun(
          [this, &holder](){
            checkpoint_proxy(holder, checkpoint_epoch);

            auto thread_id = vt::sched::ThreadAction::getActiveThreadID();
            vt::theSched()->enqueue([thread_id](){
              vt::theSched()->getThreadManager()->
                deallocateThread(thread_id);
            });
          });
        });
      }
    });
  }
  
  vt::theTerm()->popEpoch(checkpoint_epoch);
  vt::theTerm()->finishedEpoch(checkpoint_epoch);
}

void VTContext::checkpoint_proxy(ProxyHolder& holder, vt::EpochType epoch){
  vt::theTerm()->pushEpoch(epoch);

  for(auto& dep : holder.deps()){
    auto* dep_holder_ptr = holders[dep.first];
    assert(dep_holder_ptr != nullptr);
    auto& dep_holder = *dep_holder_ptr;

    dep.second = dep_holder.checkpointed_version();

    if(holder.checkpointed_version() != dep_holder.checkpointed_version())
      fmt::print("Warning: kr:: {}@{} depends on {}@{}\n", holder, holder.checkpointed_version(), 
                 dep_holder, dep_holder.checkpointed_version());
  }

  const bool is_global = holder.is_element();
  //Individual elements are checkpointed once, globally.
  //Collections have their dependency info checkpointed per-rank
  m_backend->checkpoint(
    holder.label(),
    holder.checkpointed_version(),
      {holder.metadata_registration(), holder.data_registration()}, 
      is_global
  );

  vt::theTerm()->popEpoch(epoch);
  vt::theTerm()->releaseLocalDependency(epoch);
}

 
void VTContext::restart_proxies(const std::string& region_label, const int version) {
  //Recover anything marked with a checkpointed_version set during
  //  the default member recovery step, and traverse dependencies.
  //We send all messages required for configuring version info,
  //  then actual restarts are handled as actions attached to the
  //  operation epoch's finish.

  const std::string label = fmt::format("Restart {}@{}", region_label, version);
  vt::runInEpochCollective( label, [&]{
    for(auto& holder_iter : holders.map()){
      auto& holder = holder_iter.second;
        
      if(holder.checkpointed_version() > holder.restarted_version()) {
        holder.restarted_version(holder.checkpointed_version());
      }
    }
  });
}

}
