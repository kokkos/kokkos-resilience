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

#ifndef INC_KOKKOS_RESILIENCE_CONTEXT_VT_COMMON_HPP
#define INC_KOKKOS_RESILIENCE_CONTEXT_VT_COMMON_HPP

#include <string_view>

#include <vt/vt.h>
#include "resilience/util/VTUtil.hpp"

//#define VTCONTEXT_LOG_EVENTS

namespace KokkosResilience::Context::VT {
  using namespace KokkosResilience::Util::VT;

  //Actions available through the VTContext action handler
  //Not all actions are valid for all proxy types.
  //Use macros to automatically generate to_string
#define KR_VT_PROXY_ACTIONS(f) \
    f(GET_HOLDER_AT),\
    f(FETCH_STATUS),\
    f(SET_STATUS),\
    f(SET_TRACKED),\
    f(SET_CHECKPOINTED_VERSION),\
    f(SET_RESTARTED_VERSION),\
    f(MODIFY),\
    f(REGISTER),\
    f(DEREGISTER),\
    f(CHECK_LOCAL),\
    f(CHECK_DYNAMIC),\
    f(CHECK_MISSING),\
    f(DEREGISTER_EVENT_LISTENER),\
    f(MIGRATE_STATUS)

#define KR_VT_ENUM_LIST(x) x
#define KR_VT_ENUM_LIST_STR(x) #x 

  enum ProxyAction {
    KR_VT_PROXY_ACTIONS(KR_VT_ENUM_LIST)
  };
  
  //Information about checkpoint/recovery state
  struct ProxyStatus;

  //Untyped holder with re-typing capabilities.
  //Holds ProxyStatus, manages access.
  class ProxyHolder;
  
  class ProxyMap;
  
  class VTContext;
  using VTContextProxy = VTObj<VTContext>;
  using VTContextElmProxy = VTObjElm<VTContext>;
}

#endif
