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
#ifndef INC_RESILIENCE_BACKEND_AUTOMATICBASE_HPP
#define INC_RESILIENCE_BACKEND_AUTOMATICBASE_HPP

#include "resilience/registration/Registration.hpp"
#include <unordered_set>
#include <memory>

namespace KokkosResilience{

//Avoiding cyclic dependency.
class ContextBase;

class AutomaticBackendBase {
public:
  explicit AutomaticBackendBase(ContextBase* ctx) : m_context(ctx) {};

  virtual ~AutomaticBackendBase() = default;
  
  virtual void register_member(Registration& member) = 0;

  virtual void register_members(std::unordered_set<Registration>& members){
    for(auto member : members) register_member(member);
  }

  virtual bool checkpoint(const std::string& label, int version, 
                          const std::unordered_set<Registration> &members) = 0;
  
 

  virtual int latest_version(const std::string& label) const noexcept = 0;

  virtual bool restart_available(const std::string& label, int version){
    return latest_version(label) == version; 
  };

  virtual bool restart(const std::string& label, int verison, 
                       const std::unordered_set<Registration> &members) = 0;
  
  
  virtual void reset() = 0;


  ContextBase* const m_context;
  
  
  //Delete potentially problematic functions for maintaining consistent state
  AutomaticBackendBase(const AutomaticBackendBase&) = delete;
  AutomaticBackendBase(AutomaticBackendBase&&) noexcept = delete;
  AutomaticBackendBase &operator=( const AutomaticBackendBase & ) = delete;
  AutomaticBackendBase &operator=( AutomaticBackendBase && ) = default;
};
  
using AutomaticBackend = std::shared_ptr<AutomaticBackendBase>;
}


#endif
