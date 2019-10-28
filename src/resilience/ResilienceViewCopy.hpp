

#ifndef __RESILIENCE_VIEW_COPY_H
#define __RESILIENCE_VIEW_COPY_H

#include <Kokkos_CopyViews.hpp>
#include <impl/MirrorTracker.hpp>

namespace Kokkos {

// Create a mirror in a new space (specialization for different space)
template <class Space, class T, class... P>
typename Impl::MirrorType<Space, T, P...>::view_type create_chkpt_mirror(
    const Space&, const Kokkos::View<T, P...>& src,
    typename std::enable_if<std::is_same<
        typename ViewTraits<T, P...>::specialize, void>::value>::type* = 0) {
  typedef
      typename Impl::MirrorType<Space, T, P...>::view_type chkpt_mirror_type;
  std::string sLabel = src.label();
  if (sLabel.length() == 0) {
    char sTemp[32];
    sprintf(sTemp, "view_%08x", (unsigned long)src.data());
    sLabel = sTemp;
    printf(
        "WARNING: creating checkpoint mirror without a label...generating auto "
        "label: %s \n",
        sLabel.c_str());
  }
  chkpt_mirror_type chkpt(sLabel, src.layout());
  KokkosResilience::MirrorTracker::track_mirror(
      Space::name(), sLabel, chkpt.data(), src.data());

  return chkpt;
}

}

#endif
