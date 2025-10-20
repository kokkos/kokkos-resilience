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

#ifndef INC_RESILIENCE_BACKEND_BASE_HPP
#define INC_RESILIENCE_BACKEND_BASE_HPP

#include "resilience/registration/Registration.hpp"
#include <unordered_set>
#include <memory>

//Avoiding cyclic dependency.
namespace KokkosResilience {
  class ContextBase;
}

namespace KokkosResilience::Impl::BackendImpl {
  class Base {
  public:
    using Members = std::unordered_set<Registration>;

    explicit Base(ContextBase& ctx) : m_context(ctx) {};

    virtual ~Base() = default;

    //All members should be registered before being checkpointed or restarted
    virtual void register_member(Registration member) {};
    virtual void deregister_member(Registration member) {};

    //as_global to checkpoint indepently of process (PID)
    virtual bool checkpoint(
      const std::string& label, int version, const Members& members
    ) = 0;

    //Get the highest version available which is still less than max
    //  (or just the highest, if max=0)
    virtual int latest_version(const std::string& label, int max = 0) const = 0;

    //Returns failure flag for recovering the specified members.
    //as_global to restart independently of PID
    virtual bool restart(
      const std::string& label, int version, const Members& members
    ) = 0;

    //Reset any state, allowing for compatibility with a new process joining
    virtual void reset() = 0;

    virtual bool restart_available(const std::string& label, int version){
std::cerr << "latest restart (max = " << version+1 << ") = " << latest_version(label, version+1) << std::endl;
      return latest_version(label, version+1) == version;
    };

    ContextBase& m_context;

    //Delete potentially problematic functions for maintaining consistent state
    Base(const Base&) = delete;
    Base &operator=(const Base &) = delete;

    Base(Base&&) noexcept = default;
    Base &operator=(Base &&) = default;
  };
}

namespace KokkosResilience {
  using Backend = std::shared_ptr<Impl::BackendImpl::Base>;

  namespace Impl {
    Backend make_backend(ContextBase& ctx);
  }
}


#endif
