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
#include <cassert>
#include <fstream>
#include <exception>

#include "StdFileBackend.hpp"
#include "resilience/context/ContextBase.hpp"
#include "resilience/util/Trace.hpp"

namespace KokkosResilience {

StdFileBackend::StdFileBackend(ContextBase& ctx)
    : AutomaticBackendBase(ctx) {
  
  auto config = m_context.config()["backends"]["stdfile"];
  auto config_dir = config.get("directory");
  auto config_prefix = config.get("filename_prefix");

  if(config_dir) {
    checkpoint_dir = config_dir->template as<std::string>();
    
    using namespace std::filesystem;

    std::error_code err;
    create_directories(checkpoint_dir, err);
    if(err) throw ConfigValueError("backends:stdfile:directory invalid " + err.message());

    if(status(checkpoint_dir).type() != file_type::directory){
      throw ConfigValueError("backends:stdfile:directory not actually a directory");
    }
  }

  if(config_prefix) {
    checkpoint_prefix = config_prefix->template as<std::string>();
  }
}



bool StdFileBackend::checkpoint(
    const std::string &label, int version,
    const std::unordered_set<Registration>& members,
    bool as_global) {

  //Files have a header: <file size> <Member0 ID> <Member0 offset> ... <MemberN ID> <MemberN offset>
  //Header lasts until Member0 offset
  const size_t header_size = sizeof(size_t) + (sizeof(int)+sizeof(size_t))*members.size();

  std::vector<int> member_hashes(members.size());
  std::vector<size_t> member_offsets(members.size());
  
  auto filename = checkpoint_file(label, version, as_global);

  auto write_trace = Util::begin_trace<Util::TimingTrace>(m_context, "write");
  bool success = true;
  //try {
    std::ofstream file(filename, std::ios::binary);
    file.seekp(header_size);

    size_t index = 0;
    for (auto& member : members) {
      member_hashes[index] = member.hash();
      member_offsets[index] = file.tellp();

      if(!member->serializer()(file)){
        fprintf(stderr, "Warning: In checkpoint of %s version %d, member %s serialization failed! Stream bits: good:%d fail:%d bad:%d eof:%d \n", label.c_str(), version, member->name.c_str(), file.good(), file.fail(), file.bad(), file.eof());
        success = false;
      }
      index++;
    }

    size_t full_size = file.tellp();

    file.seekp(0);
    file.write((char*) &full_size, sizeof(full_size));

    for(index = 0; index < members.size(); index++){
      file.write((char*) &member_hashes[index], sizeof(int));
      file.write((char*) &member_offsets[index], sizeof(size_t));
    }

    latest_versions[label] = version;
  //} catch (std::exception& e) {
  //  fprintf(stderr, "Error checkpointing region %s version %d to file %s: %s\n",
  //          label.c_str(), version, std::string(filename).c_str(), e.what());
  //  success = false;
  //}
  write_trace.end();
  return success;
}


std::filesystem::path StdFileBackend::checkpoint_file(
    const std::string& label, int version, bool as_global) const {
  return checkpoint_dir / (checkpoint_prefix + label + 
                            ( as_global ? "" : "." + std::to_string(m_context.m_pid) ) + 
                          "." + std::to_string(version));
}

bool StdFileBackend::restart_available(const std::string &label, int version, bool as_global) {
  return std::filesystem::exists(checkpoint_file(label, version, as_global));
}

int StdFileBackend::latest_version(const std::string &label, int max, bool as_global) const noexcept {
 
  auto iter = latest_versions.find(label);
  if(iter != latest_versions.end() && (max == 0 || iter->second < max)) return iter->second;

  int result = -1;
  bool successful = false;
  try {
    std::string basename = checkpoint_file(label, 0, as_global).filename().stem();

    for(auto& dir_entry : std::filesystem::directory_iterator(checkpoint_dir)){
      path filename = dir_entry.path().filename();

      if(filename.stem() != basename) continue;

      std::string file_ext = filename.extension();
      if(file_ext.size() < 2) continue;

      //Remove the dot from, eg, ".100"
      file_ext.erase(0,1);
      try {
        int version = stoi(file_ext);
        if(max == 0 || version < max) {
          result = result < version ? version : result;
         }
      } catch(...) {}
    }

    successful = true;
  } catch(...) {}

  if(max == 0 && successful) latest_versions[label] = result;

  return result;
}



struct FileMember {
  size_t start, stop;
  const Registration* registration = nullptr;
};

//We want to read through the file in-order where possible,
//so we build an ordered vector representing which registration to 
//restore to as we go.
//File path used for providing error context.
std::vector<FileMember>
read_header(std::istream& file, const std::unordered_set<Registration>& registrations, std::filesystem::path filename){
  std::vector<FileMember> members;
  std::unordered_map<int, int> hash_to_member;

  size_t file_size;
  file.read((char*) &file_size, sizeof(size_t));

  size_t header_size = file_size; //temporary estimate
  while(size_t(file.tellg()) < header_size){
    members.push_back(FileMember());
    int idx = members.size() - 1;

    int hash; size_t start;
    file.read((char*) &hash, sizeof(int));
    file.read((char*) &start, sizeof(size_t));

    members[idx].start = start;
    if(idx > 0) members[idx-1].stop = start;
    
    hash_to_member[hash] = idx;

    //Fix estimate after we pull the first member info.
    if(header_size == file_size){
      header_size = start;
    }
  }
  if(members.empty()){
    fprintf(stderr, "No members found in file %s but %lu expected.\n", std::string(filename).c_str(), registrations.size());
    return {};
  } else {
    members.back().stop = file_size;
  }
  
  for(auto& reg : registrations){
    auto iter = hash_to_member.find(reg.hash());
    if(iter == hash_to_member.end()){
      fprintf(stderr, "Warning: Checkpoint is missing member %s!\n", reg->name.c_str());
    }

    int idx = iter->second;
    members[idx].registration = &reg;
  }

  return members;
}

bool StdFileBackend::restart(
    const std::string &label, int version,
    const std::unordered_set<Registration>& registrations,
    bool as_global) {
  auto read_trace = Util::begin_trace<Util::TimingTrace>(m_context, "read");
  if(registrations.empty()){
    read_trace.end();
    return true;
  }

  try {
    auto filename = checkpoint_file(label, version, as_global);
    std::ifstream file(filename, std::ios::binary);
    
    auto file_members = read_header(file, registrations, filename);
  
    for(auto& member : file_members){
      if(member.registration == nullptr) continue;

      file.seekg(member.start);
      
      const Registration& reg = *(member.registration);
      if(!reg->deserializer()(file)){
        fprintf(stderr, "Warning: In restart of %s version %d, member %s deserialization failed!\n", label.c_str(), version, reg->name.c_str()); 
      }
      
      size_t actual_stop = file.tellg();
      if(actual_stop != member.stop){
        bool more = actual_stop > member.stop;
        size_t amt = more ? actual_stop-member.stop : member.stop-actual_stop;
        fprintf(stderr, "Warning: In restart of %s version %d, member %s deserialized with %lu %s bytes than in the checkpoint!\n",
            label.c_str(), version, reg->name.c_str(), amt, more ? "more" : "fewer");
      }
    }
  } catch (...) {
    read_trace.end();
    return false;
  }

  read_trace.end();
  return true;
}

}  // namespace KokkosResilience
