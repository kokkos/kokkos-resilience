
#include <Kokkos_Core.hpp>

#include <impl/MirrorTracker.hpp>

namespace KokkosResilience {

MirrorTracker* MirrorTracker::mirror_list = nullptr;

MirrorTracker* MirrorTracker::get_filtered_mirror_list(
    const std::string mem_space) {
  MirrorTracker* return_list = nullptr;
  if (mirror_list == nullptr) {
    //    printf("get_filtered mirror list --- ah there's no list...\n");
  }
  MirrorTracker* pSrch = mirror_list;
  while (pSrch != nullptr) {
    // printf("searching list: %s, %s \n", pSrch->mem_space_name.c_str(),
    // pSrch->label.c_str());
    if (pSrch->mem_space_name == mem_space) {
      MirrorTracker* pNew = new MirrorTracker(*pSrch);
      if (return_list == nullptr) {
        return_list = pNew;
      } else {
        pNew->pNext        = return_list;
        return_list->pPrev = pNew;
        return_list        = pNew;
      }
    }
    pSrch = pSrch->pNext;
  }

  return return_list;
}

MirrorTracker* MirrorTracker::get_filtered_mirror_entry(
    const std::string mem_space, const std::string lbl) {
  MirrorTracker* return_entry = nullptr;
  MirrorTracker* pSrch        = mirror_list;
  while (pSrch != nullptr) {
    if (pSrch->mem_space_name == mem_space && pSrch->label == lbl) {
      return_entry = new MirrorTracker(*pSrch);
      break;
    }
    pSrch = pSrch->pNext;
  }
  return return_entry;
}
// note that the dst_ and src_pointers are pointing to the data() element of the
// record...need to extract the record pointer from that
void MirrorTracker::track_mirror(
    const std::string mem_space, const std::string lbl, void* dst_,
    void* src_) {
  using header_type = Kokkos::Impl::SharedAllocationHeader;
  MirrorTracker* pTrack = nullptr;
  MirrorTracker* pSrch  = mirror_list;
  while (pSrch != nullptr) {
    if (pSrch->label == lbl) {
      pTrack = pSrch;
      break;
    }
    pSrch = pSrch->pNext;
  }
  bool bAddToList = false;
  if (pTrack == nullptr) {
    pTrack        = new MirrorTracker();
    pTrack->label = lbl;
    bAddToList    = true;
  }
  header_type* pDstHeader = ((reinterpret_cast<header_type*>(dst_)) - 1);
  pTrack->dst                        = pDstHeader->get_record();
  header_type* pSrcHeader = ((reinterpret_cast<header_type*>(src_)) - 1);
  pTrack->src                        = pSrcHeader->get_record();
  pTrack->mem_space_name             = mem_space;

  if (bAddToList) {
    // JSM TODO need some type of locking mechanism here ...
    if (mirror_list == nullptr) {
      //      printf("initializing new list: %s, %s \n",
      //      pTrack->mem_space_name.c_str(), pTrack->label.c_str());
      mirror_list = pTrack;
    } else {
      //      printf("inserting into list: %s, %s \n",
      //      pTrack->mem_space_name.c_str(), pTrack->label.c_str());
      pTrack->pNext      = mirror_list;
      mirror_list->pPrev = pTrack;
      mirror_list        = pTrack;
    }
  }
}

void MirrorTracker::release_mirror(void* dst_) {
  MirrorTracker* pSrch = mirror_list;
  while (pSrch != nullptr) {
    if (pSrch->dst == dst_ || pSrch->src == dst_) {
      // JSM TODO need some type of locking mechanism here ...
      if (pSrch->pNext != nullptr) pSrch->pNext->pPrev = pSrch->pPrev;
      if (pSrch->pPrev != nullptr) pSrch->pPrev->pNext = pSrch->pNext;
      delete pSrch;
      break;
    }
    pSrch = pSrch->pNext;
  }
}

void MirrorTracker::release_mirror(const std::string lbl) {
  MirrorTracker* pSrch = mirror_list;
  while (pSrch != nullptr) {
    if (pSrch->label == lbl) {
      // JSM TODO need some type of locking mechanism here ...
      if (pSrch->pNext != nullptr) pSrch->pNext->pPrev = pSrch->pPrev;
      if (pSrch->pPrev != nullptr) pSrch->pPrev->pNext = pSrch->pNext;
      delete pSrch;
      break;
    }
    pSrch = pSrch->pNext;
  }
}

} // KokkosResilience
