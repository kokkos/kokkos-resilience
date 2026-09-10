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
#include "FenixBackend.hpp"

#include <sstream>
#include <boost/iostreams/device/array.hpp>
#include <boost/iostreams/device/back_inserter.hpp>
#include <boost/iostreams/stream.hpp>

#include <fenix.h>

#define FENIX_SAFE_CALL(call) \
  do { \
    try { \
      KokkosResilience::fenix_safe_call(call, #call, __FILE__, __LINE__); \
    } catch (const fenix::RuntimeException &e) { \
      KokkosResilience::fenix_safe_call(e.error, #call, __FILE__, __LINE__); \
    } \
  } while(false)

namespace KokkosResilience {

namespace {

void fenix_throw(const std::string& msg) {
  Kokkos::Impl::throw_runtime_exception(std::string("KokkosResilience::FenixMemoryBackend::") + msg);
}

void fenix_safe_call(int status, const char* call, const char* file, int line) {
  if (status != FENIX_SUCCESS) {
    std::ostringstream msg;
    msg << "[fenix error] " << file << ":" << line << " " << call;
    fenix_throw(msg.str());
  }
}

void fenix_create_data_group(MPI_Comm mpi_comm, int group_id) {
  int mpi_size;
  MPI_Comm_size(mpi_comm, &mpi_size);

  const int start_time_stamp = 0;
  const int checkpoint_depth = 0;
  const int policy_name      = FENIX_DATA_POLICY_IN_MEMORY_RAID;
  int policy_value[3]        = {1, std::max(1, mpi_size / 2), 0};

  int flag;
  FENIX_SAFE_CALL(Fenix_Data_group_create(group_id, mpi_comm, start_time_stamp, checkpoint_depth, policy_name,
                                          policy_value, &flag));
}

Registration unalias_member(const std::unordered_map<std::string, Registration>& alias_map,
                            const Registration& member) {
  auto alias_iter = alias_map.find(member->name);
  if (alias_iter != alias_map.end()) {
    return unalias_member(alias_map, alias_iter->second);
  }
  return member;
}

std::unordered_set<Registration> get_unaliased_member_list(
    const std::unordered_map<std::string, Registration>& alias_map, const std::unordered_set<Registration>& members) {
  std::unordered_set<Registration> unaliased_members;
  for (auto&& member : members) {
    unaliased_members.insert(unalias_member(alias_map, member));
  }
  return unaliased_members;
}

}  // namespace

FenixMemoryBackend::FenixMemoryBackend(ContextBase& ctx, MPI_Comm mpi_comm) : m_context(&ctx), m_mpi_comm(mpi_comm) {}

FenixMemoryBackend::~FenixMemoryBackend() {
  clear_checkpoints();
  reset();
}

void FenixMemoryBackend::checkpoint(const std::string& label, int version,
                                    const std::unordered_set<Registration>& members) {
  const int group_id = static_cast<int>(label_hash(label));

  if (not Fenix_Data_group_created(group_id)) {
    fenix_create_data_group(m_mpi_comm, group_id);
    m_group_ids.emplace(group_id);
  }

  // store version information in the checkpoint
  FENIX_SAFE_CALL(Fenix_Data_member_define(group_id, member_id_of_version, &version, sizeof(int), MPI_CHAR));
  FENIX_SAFE_CALL(Fenix_Data_member_store(group_id, member_id_of_version, FENIX_DATA_SUBSET_FULL));

  auto unaliased_members = get_unaliased_member_list(m_alias_map, members);

  // store actual members alongside their size information
  for (auto&& member : unaliased_members) {
    std::vector<char> buffer;
    auto sink = boost::iostreams::back_inserter(buffer);
    boost::iostreams::stream<decltype(sink)> stream(sink);

    member->serialize(stream);
    stream.flush();

    char* data = buffer.data();
    int length = buffer.size();

    const int member_hash = static_cast<int>(member->hash());

    const int length_id = member_id_offset + 2 * member_hash;
    FENIX_SAFE_CALL(Fenix_Data_member_define(group_id, length_id, &length, sizeof(int), MPI_CHAR));
    FENIX_SAFE_CALL(Fenix_Data_member_store(group_id, length_id, FENIX_DATA_SUBSET_FULL));

    const int member_id = length_id + 1;
    FENIX_SAFE_CALL(Fenix_Data_member_define(group_id, member_id, data, length, MPI_CHAR));
    FENIX_SAFE_CALL(Fenix_Data_member_store(group_id, member_id, FENIX_DATA_SUBSET_FULL));
  }

  int time_stamp;
  FENIX_SAFE_CALL(Fenix_Data_commit(group_id, &time_stamp));

  m_latest_version[label] = version;
}

void FenixMemoryBackend::restart(const std::string& label, int version, std::unordered_set<Registration>& members) {
  const int group_id = label_hash(label);

  if (not Fenix_Data_group_created(group_id)) {
    fenix_throw("restart(): data group does not exist");
  }

  if (not Fenix_Data_member_created(group_id, member_id_of_version)) {
    fenix_throw("restart(): data member does not exist");
  }

  int time_stamp;
  {
    int num_snapshot;
    FENIX_SAFE_CALL(Fenix_Data_group_get_number_of_snapshots(group_id, &num_snapshot));

    if (num_snapshot == 0) {
      fenix_throw("restart(): data group does not contain any snapshots");
    }

    int position = 0;
    while (position < num_snapshot) {
      FENIX_SAFE_CALL(Fenix_Data_group_get_snapshot_at_position(group_id, position, &time_stamp));

      int current_version;
      FENIX_SAFE_CALL(
          Fenix_Data_member_restore(group_id, member_id_of_version, &current_version, sizeof(int), time_stamp, NULL));

      if (current_version == version) {
        break;
      }

      ++position;
    }

    if (position == num_snapshot) {
      fenix_throw("restart(): requested version not found");
    }
  }

  auto unaliased_members = get_unaliased_member_list(m_alias_map, members);

  for (auto&& member : unaliased_members) {
    const int member_hash = static_cast<int>(member->hash());

    const int length_id = member_id_offset + 2 * member_hash;
    const int member_id = length_id + 1;

    if (not Fenix_Data_member_created(group_id, length_id)) {
      fenix_throw("restart(): data member does not exist");
    }

    if (not Fenix_Data_member_created(group_id, member_id)) {
      fenix_throw("restart(): data member does not exist");
    }

    int length;
    FENIX_SAFE_CALL(Fenix_Data_member_restore(group_id, length_id, &length, sizeof(int), time_stamp, NULL));

    std::vector<char> buffer(length);
    FENIX_SAFE_CALL(Fenix_Data_member_restore(group_id, member_id, buffer.data(), length, time_stamp, NULL));

    boost::iostreams::array_source source(buffer.data(), buffer.size());
    boost::iostreams::stream<decltype(source)> stream(source);

    member->deserialize(stream);
  }
}

int FenixMemoryBackend::latest_version(const std::string& label) const noexcept {
  auto iter = m_latest_version.find(label);
  if (iter != m_latest_version.end()) {
    return iter->second;
  }

  const int group_id = label_hash(label);

  if (not Fenix_Data_group_created(group_id)) {
    return -1;
  }

  if (not Fenix_Data_member_created(group_id, member_id_of_version)) {
    return -1;
  }

  const int position = 0;  // latest snapshot is at position 0
  int time_stamp;
  FENIX_SAFE_CALL(Fenix_Data_group_get_snapshot_at_position(group_id, position, &time_stamp));

  int version;
  FENIX_SAFE_CALL(Fenix_Data_member_restore(group_id, member_id_of_version, &version, sizeof(int), time_stamp, NULL));

  m_latest_version[label] = version;

  return version;
}

bool FenixMemoryBackend::restart_available(const std::string& label, int version) {
  return version == latest_version(label);
}

void FenixMemoryBackend::clear_checkpoints() {
  for (const auto& group_id : m_group_ids) {
    int num_member;
    FENIX_SAFE_CALL(Fenix_Data_group_get_number_of_members(group_id, &num_member));

    for (int i = 0; i < num_member; ++i) {
      int member_id;
      int position = 0;
      FENIX_SAFE_CALL(Fenix_Data_group_get_member_at_position(group_id, &member_id, position));
      Fenix_Data_member_delete(group_id, member_id);
    }

    // not deleteting data group to avoid double free in fenix
  }
}

void FenixMemoryBackend::reset() {
  m_latest_version.clear();
  m_alias_map.clear();
}

void FenixMemoryBackend::register_alias(Registration& member, const std::string& alias) {
  m_alias_map.try_emplace(alias, member);
}

}  // namespace KokkosResilience
