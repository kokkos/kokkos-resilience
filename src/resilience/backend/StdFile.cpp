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

#include "resilience/backend/StdFile.hpp"
#include "resilience/context/Context.hpp"
#include "resilience/util/Trace.hpp"

namespace KokkosResilience::Impl::BackendImpl {

StdFile::StdFile(ContextBase& ctx)
  : Base(ctx) {
  
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



bool StdFile::checkpoint(
  const std::string& label, int version, const Members& members
) {
std::cerr << "Checkpointing " << members.size() << " members to " << label << std::endl;
  auto write_trace =
    Util::begin_trace<Util::TimingTrace<std::string>>(m_context, "write");

  bool success = write_to_file(checkpoint_file(label, version), members);
  if(success) latest_versions[label] = version;

  write_trace.end();
  return success;
}

bool StdFile::restart(
  const std::string& label, int version, const Members& members
) {
  auto read_trace = Util::begin_trace<Util::TimingTrace<std::string>>(
    m_context, "StdFile::restart_from_file"
  );

  auto filename = checkpoint_file(label, version);
  bool success = read_from_file(checkpoint_file(label, version), members);

  read_trace.end();
  return success;
}


std::filesystem::path StdFile::checkpoint_file(
  const std::string& label, int version
) const {
  return checkpoint_dir / (checkpoint_prefix + label + "." +
    std::to_string(m_context.pid() ) + "." + std::to_string(version));
}

bool StdFile::restart_available(const std::string& label, int version) {
  return std::filesystem::exists(checkpoint_file(label, version));
}

int StdFile::latest_version(const std::string& label, int max) const {
  auto iter = latest_versions.find(std::string(label));
  if(iter != latest_versions.end() && (max == 0 || iter->second < max))
    return iter->second;

  int result = -1;
  bool successful = false;
  try {
    std::string basename = checkpoint_file(label, 0).filename().stem();

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

bool StdFile::write_to_file(
  path filename, const Members& members
) noexcept {
  //Files have a header: 
  //  <file size><Member0 ID><Member0 offset>...<MemberN ID><MemberN offset>
  //Header lasts until Member0 offset
  const size_t header_size = sizeof(size_t) + (sizeof(int)+sizeof(size_t))*members.size();

  std::vector<int> member_hashes(members.size());
  std::vector<size_t> member_offsets(members.size());

  bool success = true;
  try {
    std::ofstream file(filename, std::ios::binary);
    file.seekp(header_size);

    size_t index = 0;
    for (auto& member : members) {
      member_hashes[index] = member.hash();
      member_offsets[index] = file.tellp();

      if(!member->serializer()(file)){
        fprintf(stderr, "Warning: In checkpoint to file %s, member %s "
          "serialization failed! Stream bits: good:%d fail:%d bad:%d eof:%d \n",
          filename.string().c_str(), member->name.c_str(), file.good(),
          file.fail(), file.bad(), file.eof()
        );
        success = false;
      }
std::cerr << "Writing member " << member->name << " (" << member.hash() << ") to offset " << member_offsets[index] << std::endl;
      index++;
    }

    size_t full_size = file.tellp();

    file.seekp(0);
    file.write((char*) &full_size, sizeof(full_size));

    for(index = 0; index < members.size(); index++){
      file.write((char*) &member_hashes[index], sizeof(int));
      file.write((char*) &member_offsets[index], sizeof(size_t));
    }
  } catch (std::exception& e) {
    fprintf(stderr, "Error checkpointing to file %s: %s\n",
      std::string(filename).c_str(), e.what());
    success = false;
  }
  return success;
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

  size_t header_size = file_size; //temporary estimate to safely end loop
  while(size_t(file.tellg()) < header_size){
    members.push_back(FileMember());
    int idx = members.size() - 1;

    int hash; size_t start;
    file.read((char*) &hash, sizeof(int));
    file.read((char*) &start, sizeof(size_t));

    members[idx].start = start;
    if(idx > 0) members[idx-1].stop = start;
    
std::cerr << "Reading member header (" << hash << ") has offset " << start << std::endl;

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

bool StdFile::read_from_file(path filename, const Members& members) noexcept {
  if(members.empty()){
    return true;
  }

  bool success = true;
  try {
    std::ifstream file(filename, std::ios::binary);
    
    auto file_members = read_header(file, members, filename);
  
    for(auto& member : file_members){
std::cerr << "Member header at " << member.start << "-" << member.stop << " has registration ptr " << member.registration << std::endl;
      if(member.registration == nullptr) continue;

      file.seekg(member.start);
      
      const Registration& reg = *(member.registration);
std::cerr << "Reading member " << reg->name << " (" << reg.hash() << ") at offset " << member.start << std::endl;
      if(!reg->deserializer()(file)) fprintf(stderr,
        "Warning: In restart from file %s, member %s deserialization failed!\n",
        filename.string().c_str(), reg->name.c_str()
      ); 
      
      size_t actual_stop = file.tellg();
      if(actual_stop != member.stop){
        bool more = actual_stop > member.stop;
        size_t amt = more ? actual_stop-member.stop : member.stop-actual_stop;
        fprintf(stderr, "Warning: In restart from file %s, member %s "
          "deserialized with %lu %s bytes than in the checkpoint!\n",
          filename.string().c_str(), reg->name.c_str(), amt,
          more ? "more" : "fewer"
        );
      }
    }
  } catch (...) {
    success = false;
  }
  return success;
}

}  // namespace KokkosResilience
