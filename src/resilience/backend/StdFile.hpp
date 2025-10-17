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
#ifndef INC_RESILIENCE_BACKEND_STDFILE_HPP
#define INC_RESILIENCE_BACKEND_STDFILE_HPP

#include "resilience/backend/Backend.hpp"

#include <filesystem>
#include <unordered_map>

namespace KokkosResilience::Impl::BackendImpl {
  class StdFile : public Base {
   public:
    using path = std::filesystem::path;

    StdFile(ContextBase& ctx);

    //No state to manage
    void register_member(Registration) override {};
    void deregister_member(Registration) override {};

    bool checkpoint(
      const std::string& label, int version, const Members& members
    ) override;

    int latest_version(const std::string& label, int max) const override;

    bool restart_available(const std::string& label, int version) override;

    bool restart(
      const std::string& label, int version, const Members& members
    ) override;

    //No state to reset
    void reset() override {};

    //Exposed to reuse logic for VeloCFile. Returns success
    static bool write_to_file(path filename, const Members& members) noexcept;
    static bool read_from_file(path filename, const Members& members) noexcept;

   protected:
    path checkpoint_dir = "./";
    std::string checkpoint_prefix = "kr_chkpt_";

    mutable std::unordered_map<std::string, int> latest_versions;

    //The file to checkpoint/recover with
    path checkpoint_file(const std::string& label, int version) const;
  };
}  // namespace KokkosResilience

#endif  // INC_RESILIENCE_BACKEND_STDFILEBACKEND_HPP
