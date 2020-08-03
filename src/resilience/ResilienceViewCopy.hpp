

#ifndef __RESILIENCE_VIEW_COPY_H
#define __RESILIENCE_VIEW_COPY_H

#include <Kokkos_CopyViews.hpp>
#include <impl/MirrorTracker.hpp>
#include <impl/FileSpaceTypes.hpp>

namespace KokkosResilience {
   struct FileSpaceSpecializeTag{};
}

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


// need to specialize ViewMapping for File Spaces...
template <class... Prop>
struct ViewTraits<void, KokkosResilience::StdFileSpace, Prop...> {
  // Specify Space, memory traits should be the only subsequent argument.

  static_assert(
      std::is_same<typename ViewTraits<void, Prop...>::execution_space,
                   void>::value &&
          std::is_same<typename ViewTraits<void, Prop...>::memory_space,
                       void>::value &&
          std::is_same<typename ViewTraits<void, Prop...>::HostMirrorSpace,
                       void>::value &&
          std::is_same<typename ViewTraits<void, Prop...>::array_layout,
                       void>::value,
      "Only one View Execution or Memory Space template argument");

  using execution_space = KokkosResilience::StdFileSpace::execution_space;
  using memory_space = KokkosResilience::StdFileSpace::memory_space;
  using HostMirrorSpace = Kokkos::Impl::HostMirror<KokkosResilience::StdFileSpace>::Space;
  using array_layout = typename execution_space::array_layout;
  using memory_traits = typename ViewTraits<void, Prop...>::memory_traits;
  using specialize = KokkosResilience::FileSpaceSpecializeTag;
};

namespace Impl {

template <class Traits>
class ViewMapping<Traits, KokkosResilience::FileSpaceSpecializeTag> {
private:
  template <class, class...> friend class ViewMapping;
  template <class, class...> friend class Kokkos::View;

  typedef ViewOffset<typename Traits::dimension, typename Traits::array_layout,
                     void>
      offset_type;

  typedef typename ViewDataHandle<Traits>::handle_type handle_type;

  handle_type m_impl_handle;
  offset_type m_impl_offset;

  KOKKOS_INLINE_FUNCTION
  ViewMapping(const handle_type &arg_handle, const offset_type &arg_offset)
      : m_impl_handle(arg_handle), m_impl_offset(arg_offset) {}

public:
  typedef void printable_label_typedef;
  enum { is_managed = Traits::is_managed };

  //----------------------------------------
  // Domain dimensions

  enum { Rank = Traits::dimension::rank };

  template <typename iType>
  KOKKOS_INLINE_FUNCTION constexpr size_t extent(const iType &r) const {
    return m_impl_offset.m_dim.extent(r);
  }

  KOKKOS_INLINE_FUNCTION constexpr typename Traits::array_layout
  layout() const {
    return m_impl_offset.layout();
  }

  KOKKOS_INLINE_FUNCTION constexpr size_t dimension_0() const {
    return m_impl_offset.dimension_0();
  }
  KOKKOS_INLINE_FUNCTION constexpr size_t dimension_1() const {
    return m_impl_offset.dimension_1();
  }
  KOKKOS_INLINE_FUNCTION constexpr size_t dimension_2() const {
    return m_impl_offset.dimension_2();
  }
  KOKKOS_INLINE_FUNCTION constexpr size_t dimension_3() const {
    return m_impl_offset.dimension_3();
  }
  KOKKOS_INLINE_FUNCTION constexpr size_t dimension_4() const {
    return m_impl_offset.dimension_4();
  }
  KOKKOS_INLINE_FUNCTION constexpr size_t dimension_5() const {
    return m_impl_offset.dimension_5();
  }
  KOKKOS_INLINE_FUNCTION constexpr size_t dimension_6() const {
    return m_impl_offset.dimension_6();
  }
  KOKKOS_INLINE_FUNCTION constexpr size_t dimension_7() const {
    return m_impl_offset.dimension_7();
  }

  // Is a regular layout with uniform striding for each index.
  using is_regular = typename offset_type::is_regular;

  KOKKOS_INLINE_FUNCTION constexpr size_t stride_0() const {
    return m_impl_offset.stride_0();
  }
  KOKKOS_INLINE_FUNCTION constexpr size_t stride_1() const {
    return m_impl_offset.stride_1();
  }
  KOKKOS_INLINE_FUNCTION constexpr size_t stride_2() const {
    return m_impl_offset.stride_2();
  }
  KOKKOS_INLINE_FUNCTION constexpr size_t stride_3() const {
    return m_impl_offset.stride_3();
  }
  KOKKOS_INLINE_FUNCTION constexpr size_t stride_4() const {
    return m_impl_offset.stride_4();
  }
  KOKKOS_INLINE_FUNCTION constexpr size_t stride_5() const {
    return m_impl_offset.stride_5();
  }
  KOKKOS_INLINE_FUNCTION constexpr size_t stride_6() const {
    return m_impl_offset.stride_6();
  }
  KOKKOS_INLINE_FUNCTION constexpr size_t stride_7() const {
    return m_impl_offset.stride_7();
  }

  template <typename iType>
  KOKKOS_INLINE_FUNCTION void stride(iType *const s) const {
    m_impl_offset.stride(s);
  }

  //----------------------------------------
  // Range span

  /** \brief  Span of the mapped range */
  KOKKOS_INLINE_FUNCTION constexpr size_t span() const {
    return m_impl_offset.span();
  }

  /** \brief  Is the mapped range span contiguous */
  KOKKOS_INLINE_FUNCTION constexpr bool span_is_contiguous() const {
    return m_impl_offset.span_is_contiguous();
  }

  typedef typename ViewDataHandle<Traits>::return_type reference_type;
  typedef typename Traits::value_type *pointer_type;

  /** \brief  Query raw pointer to memory */
  KOKKOS_INLINE_FUNCTION constexpr pointer_type data() const {
    return m_impl_handle;
  }

  //----------------------------------------
  // The View class performs all rank and bounds checking before
  // calling these element reference methods.

  KOKKOS_FORCEINLINE_FUNCTION
  reference_type reference() const { return m_impl_handle[0]; }

  //----------------------------------------

private:
  enum { MemorySpanMask = 8 - 1 /* Force alignment on 8 byte boundary */ };
  enum { MemorySpanSize = sizeof(typename Traits::value_type) };

public:
  /** \brief  Span, in bytes, of the referenced memory */
  KOKKOS_INLINE_FUNCTION constexpr size_t memory_span() const {
    return (m_impl_offset.span() * sizeof(typename Traits::value_type) +
            MemorySpanMask) &
           ~size_t(MemorySpanMask);
  }

  //----------------------------------------

  KOKKOS_DEFAULTED_FUNCTION ~ViewMapping() = default;
  KOKKOS_INLINE_FUNCTION ViewMapping() : m_impl_handle(), m_impl_offset() {}

  KOKKOS_DEFAULTED_FUNCTION ViewMapping(const ViewMapping&) = default;
  KOKKOS_DEFAULTED_FUNCTION ViewMapping& operator=(const ViewMapping&) =
      default;

  KOKKOS_DEFAULTED_FUNCTION ViewMapping(ViewMapping&&) = default;
  KOKKOS_DEFAULTED_FUNCTION ViewMapping& operator=(ViewMapping&&) = default;

  //----------------------------------------

  /**\brief  Span, in bytes, of the required memory */
  KOKKOS_INLINE_FUNCTION
  static constexpr size_t memory_span(
      typename Traits::array_layout const& arg_layout) {
    using padding = std::integral_constant<unsigned int, 0>;
    return (offset_type(padding(), arg_layout).span() * MemorySpanSize +
            MemorySpanMask) &
           ~size_t(MemorySpanMask);
  }

  /**\brief  Wrap a span of memory */
  template <class... P>
  KOKKOS_INLINE_FUNCTION ViewMapping(
      Kokkos::Impl::ViewCtorProp<P...> const& arg_prop,
      typename Traits::array_layout const& arg_layout)
      : m_impl_handle(
            ((Kokkos::Impl::ViewCtorProp<void, pointer_type> const&)arg_prop)
                .value),
        m_impl_offset(std::integral_constant<unsigned, 0>(), arg_layout) {}

  /**\brief  Assign data */
  KOKKOS_INLINE_FUNCTION
  void assign_data(pointer_type arg_ptr) { m_impl_handle = handle_type(arg_ptr); }

  //----------------------------------------
  /*  Allocate and construct mapped array.
   *  Allocate via shared allocation record and
   *  return that record for allocation tracking.
   */
  template <class... P>
  Kokkos::Impl::SharedAllocationRecord<>* allocate_shared(
      Kokkos::Impl::ViewCtorProp<P...> const& arg_prop,
      typename Traits::array_layout const& arg_layout) {
    using alloc_prop = Kokkos::Impl::ViewCtorProp<P...>;

    using execution_space = typename alloc_prop::execution_space;
    using memory_space    = typename Traits::memory_space;
    using value_type      = typename Traits::value_type;
    using functor_type    = ViewValueFunctor<execution_space, value_type>;
    using record_type =
        Kokkos::Impl::SharedAllocationRecord<memory_space, functor_type>;

    // Query the mapping for byte-size of allocation.
    // If padding is allowed then pass in sizeof value type
    // for padding computation.
    using padding = std::integral_constant<
        unsigned int, alloc_prop::allow_padding ? sizeof(value_type) : 0>;

    m_impl_offset = offset_type(padding(), arg_layout);

    const size_t alloc_size =
        (m_impl_offset.span() * MemorySpanSize + MemorySpanMask) &
        ~size_t(MemorySpanMask);
    const std::string& alloc_name =
        static_cast<Kokkos::Impl::ViewCtorProp<void, std::string> const&>(
            arg_prop)
            .value;
    // Create shared memory tracking record with allocate memory from the memory
    // space
    record_type* const record = record_type::allocate(
        static_cast<Kokkos::Impl::ViewCtorProp<void, memory_space> const&>(
            arg_prop)
            .value,
        alloc_name, alloc_size);

    m_impl_handle = handle_type(reinterpret_cast<pointer_type>(record->data()));

    return record;
  }
};

} // Impl namespace

//----------------------------------------------------------------------------
/** \brief  A deep copy between views of the default specialization, compatible
 * type, same non-zero rank, same contiguous layout.
 */
template <class DT, class... DP, class ST, class... SP>
inline void deep_copy(
    const View<DT, DP...>& dst, const View<ST, SP...>& src,
    typename std::enable_if<(
        (std::is_same<typename ViewTraits<DT, DP...>::specialize, 
                     KokkosResilience::FileSpaceSpecializeTag>::value ||
         std::is_same<typename ViewTraits<ST, SP...>::specialize, 
                     KokkosResilience::FileSpaceSpecializeTag>::value) &&
        (unsigned(ViewTraits<DT, DP...>::rank) != 0 ||
         unsigned(ViewTraits<ST, SP...>::rank) != 0))>::type* = nullptr) {
  using dst_type            = View<DT, DP...>;
  using src_type            = View<ST, SP...>;
  using dst_execution_space = typename dst_type::execution_space;
  using src_execution_space = typename src_type::execution_space;
  using dst_memory_space    = typename dst_type::memory_space;
  using src_memory_space    = typename src_type::memory_space;
  using dst_value_type      = typename dst_type::value_type;
  using src_value_type      = typename src_type::value_type;

  static_assert(std::is_same<typename dst_type::value_type,
                             typename dst_type::non_const_value_type>::value,
                "deep_copy requires non-const destination type");

  static_assert((unsigned(dst_type::rank) == unsigned(src_type::rank)),
                "deep_copy requires Views of equal rank");

  if (Kokkos::Profiling::profileLibraryLoaded()) {
    Kokkos::Profiling::beginDeepCopy(
        Kokkos::Profiling::make_space_handle(dst_memory_space::name()),
        dst.label(), dst.data(),
        Kokkos::Profiling::make_space_handle(src_memory_space::name()),
        src.label(), src.data(),
        src.span() * sizeof(typename dst_type::value_type));
  }

  if (dst.data() == nullptr || src.data() == nullptr) {
    // throw if dimension mismatch
    if ((src.extent(0) != dst.extent(0)) || (src.extent(1) != dst.extent(1)) ||
        (src.extent(2) != dst.extent(2)) || (src.extent(3) != dst.extent(3)) ||
        (src.extent(4) != dst.extent(4)) || (src.extent(5) != dst.extent(5)) ||
        (src.extent(6) != dst.extent(6)) || (src.extent(7) != dst.extent(7))) {
      std::string message(
          "Deprecation Error: Kokkos::deep_copy extents of views don't "
          "match: ");
      message += dst.label();
      message += "(";
      for (int r = 0; r < dst_type::Rank - 1; r++) {
        message += std::to_string(dst.extent(r));
        message += ",";
      }
      message += std::to_string(dst.extent(dst_type::Rank - 1));
      message += ") ";
      message += src.label();
      message += "(";
      for (int r = 0; r < src_type::Rank - 1; r++) {
        message += std::to_string(src.extent(r));
        message += ",";
      }
      message += std::to_string(src.extent(src_type::Rank - 1));
      message += ") ";

      Kokkos::Impl::throw_runtime_exception(message);
    }
    Kokkos::fence();
    if (Kokkos::Profiling::profileLibraryLoaded()) {
      Kokkos::Profiling::endDeepCopy();
    }
    return;
  }

  enum {
    DstExecCanAccessSrc =
        Kokkos::Impl::SpaceAccessibility<dst_execution_space,
                                         src_memory_space>::accessible
  };

  enum {
    SrcExecCanAccessDst =
        Kokkos::Impl::SpaceAccessibility<src_execution_space,
                                         dst_memory_space>::accessible
  };

  // Check for same pointers (start and end)
  dst_value_type* dst_start = dst.data();
  dst_value_type* dst_end   = dst.data() + dst.span();
  src_value_type* src_start = src.data();
  src_value_type* src_end   = src.data() + src.span();
  if (((std::ptrdiff_t)dst_start == (std::ptrdiff_t)src_start) &&
      ((std::ptrdiff_t)dst_end == (std::ptrdiff_t)src_end) &&
      (dst.span_is_contiguous() && src.span_is_contiguous())) {
    Kokkos::fence();
    if (Kokkos::Profiling::profileLibraryLoaded()) {
      Kokkos::Profiling::endDeepCopy();
    }
    return;
  }

  // Don't Check for Overlapping Views.

  // Check for same extents
  if ((src.extent(0) != dst.extent(0)) || (src.extent(1) != dst.extent(1)) ||
      (src.extent(2) != dst.extent(2)) || (src.extent(3) != dst.extent(3)) ||
      (src.extent(4) != dst.extent(4)) || (src.extent(5) != dst.extent(5)) ||
      (src.extent(6) != dst.extent(6)) || (src.extent(7) != dst.extent(7))) {
    std::string message(
        "Deprecation Error: Kokkos::deep_copy extents of views don't match: ");
    message += dst.label();
    message += "(";
    for (int r = 0; r < dst_type::Rank - 1; r++) {
      message += std::to_string(dst.extent(r));
      message += ",";
    }
    message += std::to_string(dst.extent(dst_type::Rank - 1));
    message += ") ";
    message += src.label();
    message += "(";
    for (int r = 0; r < src_type::Rank - 1; r++) {
      message += std::to_string(src.extent(r));
      message += ",";
    }
    message += std::to_string(src.extent(src_type::Rank - 1));
    message += ") ";

    Kokkos::Impl::throw_runtime_exception(message);
  }

  // If same type, equal layout, equal dimensions, equal span, and contiguous
  // memory then can byte-wise copy

  if (std::is_same<typename dst_type::value_type,
                   typename src_type::non_const_value_type>::value &&
      (std::is_same<typename dst_type::array_layout,
                    typename src_type::array_layout>::value ||
       (dst_type::rank == 1 && src_type::rank == 1)) &&
      dst.span_is_contiguous() && src.span_is_contiguous() &&
      ((dst_type::rank < 1) || (dst.stride_0() == src.stride_0())) &&
      ((dst_type::rank < 2) || (dst.stride_1() == src.stride_1())) &&
      ((dst_type::rank < 3) || (dst.stride_2() == src.stride_2())) &&
      ((dst_type::rank < 4) || (dst.stride_3() == src.stride_3())) &&
      ((dst_type::rank < 5) || (dst.stride_4() == src.stride_4())) &&
      ((dst_type::rank < 6) || (dst.stride_5() == src.stride_5())) &&
      ((dst_type::rank < 7) || (dst.stride_6() == src.stride_6())) &&
      ((dst_type::rank < 8) || (dst.stride_7() == src.stride_7()))) {
    const size_t nbytes = sizeof(typename dst_type::value_type) * dst.span();
    Kokkos::fence();
    if ((void*)dst.data() != (void*)src.data()) {
      Kokkos::Impl::DeepCopy<dst_memory_space, src_memory_space>(
          dst.data(), src.data(), nbytes);
      Kokkos::fence();
    }
  } else {
     printf("deep_copy requires contiguous Views of equal layouts\n");
  }
  if (Kokkos::Profiling::profileLibraryLoaded()) {
    Kokkos::Profiling::endDeepCopy();
  }
}
} // Kokkos namespace


#endif
