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

#include "common.hpp"
#include "ProxyHolder.hpp"
#include "VTContext.impl.hpp"
#include "ProxyHolder.impl.hpp"

namespace KokkosResilience::Context::VT {

ProxyHolder::ProxyHolder() 
 : data_reg(
     custom_registration(
       [](std::ostream& stream){return false;},
       [](std::istream& stream){return false;},
       "Invalid VTProxyHolder"
     )
   ) 
{
  assert(false && "VTProxyHolder default constructor exists for" \
                  "Magistrate serialization, but should not be used!");
}

ProxyHolder::~ProxyHolder(){
  ctx->m_backend->deregister_member(metadata_registration());
  ctx->m_backend->deregister_member(data_registration());
  if(listener_id != -1){
    invoke(DEREGISTER_EVENT_LISTENER, listener_id);
  }
}

const std::string& ProxyHolder::label() const {
  return data_reg->name;
}

std::any ProxyHolder::invoke(ProxyAction action){
  return invoker(action, nullptr);
}

bool ProxyHolder::is_local(){
  return std::any_cast<bool>( invoke(CHECK_LOCAL) );
}

bool ProxyHolder::is_dynamic(){
  return std::any_cast<bool>( invoke(CHECK_DYNAMIC) );
}

std::unordered_map<ProxyID, int>& ProxyHolder::deps(){
  wait_if_migrating();
  return _status.deps;
}

ProxyHolder* ProxyHolder::get_holder(ProxyID index_id) {
  assert(index_id.proxy_bits == this->proxy_bits);
  
  return std::any_cast<ProxyHolder*>(
      invoke(GET_HOLDER_AT, index_id.index_bits)
  );
}

vt::EpochType ProxyHolder::fetch_status(const std::string& epoch_label){
  if(fetch_epoch == vt::no_epoch){
    fetch_epoch = vt::theTerm()->makeEpochRooted(epoch_label);
    vt::theMsg()->pushEpoch(fetch_epoch);
    invoke(FETCH_STATUS);
    vt::theMsg()->popEpoch(fetch_epoch);
    vt::theTerm()->finishedEpoch(fetch_epoch);
    vt::theTerm()->addAction(fetch_epoch, [&]{
      fetch_epoch = vt::no_epoch;
    });
  }

  return fetch_epoch;
}

void ProxyHolder::tracked(bool new_tracked) {
  wait_if_migrating();
  _status.tracked = new_tracked;
  invoke(SET_TRACKED, new_tracked);
}

bool ProxyHolder::tracked() {
  wait_if_migrating();
  return _status.tracked && !_status.deleted;
}

void ProxyHolder::deleted(bool new_deleted) {
  wait_if_migrating();
  _status.deleted = new_deleted;
}

bool ProxyHolder::deleted(){
  wait_if_migrating();
  return _status.deleted;
}


void ProxyHolder::checkpointed_version(int version){
  wait_if_migrating();
  _status.checkpointed_version = version;
  invoke(SET_CHECKPOINTED_VERSION, version);
}

int ProxyHolder::checkpointed_version(){
  wait_if_migrating();
  return _status.checkpointed_version;
}


void ProxyHolder::restarted_version(int version){
  wait_if_migrating();
  invoke(SET_RESTARTED_VERSION, version);
}

int ProxyHolder::restarted_version(){
  wait_if_migrating();
  return _status.restarted_version;
}


void ProxyHolder::modified(){
  invoke(MODIFY);
}

void ProxyHolder::deregistered(std::string region_label){
  wait_if_migrating();
  invoke(DEREGISTER, region_label);
}

void ProxyHolder::registered(std::string region_label){
  wait_if_migrating();
  invoke(REGISTER, region_label);
}


void ProxyHolder::check_missing(std::unordered_set<ProxyID>* missing){
  invoke(CHECK_MISSING, missing);
}

ProxyStatus ProxyHolder::get_status(){
  wait_if_migrating();
  ProxyStatus to_ret = _status;
  to_ret.deps.clear();
  return to_ret;
}

void ProxyHolder::set_status(const ProxyStatus& new_status){
  _status.tracked = new_status.tracked;
  _status.deleted = new_status.deleted;
  _status.checkpointed_version = new_status.checkpointed_version;
  _status.restarted_version = new_status.restarted_version;
}

void ProxyHolder::migrated_out(){
  if(migrate_in_epoch != vt::no_epoch){
    //updated status info never got to us,
    //so we don't send ours and instead let
    //the prior owner's message pass on along
    vt::theTerm()->releaseLocalDependency(migrate_in_epoch);
    migrate_in_epoch = vt::no_epoch;
    return;
  }
  //Send our status info to the new location.
  invoke(MIGRATE_STATUS, _status);
}

void ProxyHolder::migrated_in(){
  if(got_status_before_migrate){
    got_status_before_migrate = false;
    return;
  }

  assert(migrate_in_epoch == vt::no_epoch);
  migrate_in_epoch = vt::theTerm()->makeEpochRooted(
    fmt::format("{} awaiting status after migrating in", *this)
  );

  vt::theTerm()->addLocalDependency(migrate_in_epoch);
  vt::theTerm()->finishedEpoch(migrate_in_epoch);
}

void ProxyHolder::migrated_status(const ProxyStatus& new_status){
  _status = new_status;
  
  if(migrate_in_epoch == vt::no_epoch){
    got_status_before_migrate = true;
  } else {
    vt::theTerm()->releaseLocalDependency(migrate_in_epoch);
    migrate_in_epoch = vt::no_epoch;
  }
}

void ProxyHolder::wait_if_migrating(){
  if(migrate_in_epoch != vt::no_epoch){
    vt::runSchedulerThrough(migrate_in_epoch);
  }
}

Registration ProxyHolder::data_registration(){
  return data_reg;
}

Registration ProxyHolder::metadata_registration(){
  using namespace checkpoint;
  return custom_registration(
    [this](std::ostream& stream) mutable {
      serializeToStream(*this, stream);
      return bool(stream);
    },
    [this](std::istream& stream) mutable {
      deserializeInPlaceFromStream(stream, this);
      return bool(stream);
    },
    label() + "meta"
  );
}

}
