

#ifndef __KOKKOS_MIRROR_TRACKER_H
#define __KOKKOS_MIRROR_TRACKER_H

#include <impl/Kokkos_SharedAlloc.hpp>

namespace KokkosResilience {

struct MirrorTracker {
  std::string label;
  void* dst;
  void* src;
  std::string mem_space_name;
  MirrorTracker* pNext;
  MirrorTracker* pPrev;

  MirrorTracker()
      : label(""),
        dst(nullptr),
        src(nullptr),
        mem_space_name(""),
        pNext(nullptr),
        pPrev(nullptr) {}

  MirrorTracker(const MirrorTracker& rhs)
      : label(rhs.label),
        dst(rhs.dst),
        src(rhs.src),
        mem_space_name(rhs.mem_space_name),
        pNext(nullptr),
        pPrev(nullptr) {}

  static void track_mirror(const std::string mem_space, const std::string lbl,
                           void* dst_, void* src_);
  static void release_mirror(void* dst);
  static void release_mirror(const std::string label);

  static MirrorTracker* get_filtered_mirror_list(const std::string mem_space);
  static MirrorTracker* get_filtered_mirror_entry(const std::string mem_space,
                                                  const std::string lbl);

  static MirrorTracker* mirror_list;

};


} //KokkosResilience

#endif
