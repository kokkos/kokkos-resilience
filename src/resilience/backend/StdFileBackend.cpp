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
#include "StdFileBackend.hpp"

#include <cassert>
#include <fstream>
#include <string>

#include <filesystem>

#include "../AutomaticCheckpoint.hpp"

#include "resilience/util/Trace.hpp"

namespace KokkosResilience {

namespace detail {

std::string versionless_filename(std::string const &filename, std::string const &label) {
  return filename + "." + label;
}

std::string full_filename(std::string const &filename, std::string const &label,
                          int version) {
  return versionless_filename(filename, label) + "." + std::to_string(version);
}
}  // namespace detail

StdFileBackend::StdFileBackend(StdFileContext<StdFileBackend> &ctx,
                               std::string const &filename)
    : m_filename(filename), m_context(ctx) {}

StdFileBackend::~StdFileBackend() = default;

void StdFileBackend::checkpoint(
    const std::string &label, int version,
    std::unordered_set<Registration> &members) {
  try {
    std::string filename = detail::full_filename(m_filename, label, version);
    std::ofstream file(filename, std::ios::binary);

    auto write_trace = Util::begin_trace(m_context, "write");
    for (auto &&member : members) {
      member->serialize(file);
    }
  } catch (...) {
  }
}

bool StdFileBackend::restart_available(const std::string &label, int version) {
  std::string filename = detail::full_filename(m_filename, label, version);
  bool exists = std::filesystem::exists(filename);
  return exists;
}

int StdFileBackend::latest_version(const std::string &label) const noexcept {
  int result = -1;
  std::string filename = detail::versionless_filename(m_filename, label);
  std::filesystem::path dir(filename);

  filename = dir.filename().string();

  dir = std::filesystem::absolute(dir).parent_path();


  for(auto &entry : std::filesystem::directory_iterator(dir)){
    if (!std::filesystem::is_regular_file(entry)) {
      continue;
    }
    if(filename == entry.path().filename().stem().string()){
        //This is a checkpoint, probably.
        try{
            int vers = std::stoi(entry.path().filename().extension().string().substr(1));
            result = std::max(result,vers);
        } catch(...) {
            //Just not the filename format we expected, could be unrelated.
        }
    }
  }
  return result;
}

void StdFileBackend::restart(
    const std::string &label, int version,
    std::unordered_set<Registration> &members) {
  try {
    std::string filename = detail::full_filename(m_filename, label, version);
    std::ifstream file(filename, std::ios::binary);

    auto read_trace = Util::begin_trace(m_context, "read");
    for (auto &&member : members) {
      member->deserialize(file);
    }
  } catch (...) {
  }
}
}  // namespace KokkosResilience
