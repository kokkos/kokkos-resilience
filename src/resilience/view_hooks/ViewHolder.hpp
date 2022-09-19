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
#ifndef INC_RESILIENCE_VIEWHOLDER_HPP
#define INC_RESILIENCE_VIEWHOLDER_HPP

#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_HostSpace.hpp>

#include <memory>
#include <type_traits>
#include <string>

namespace Kokkos {
// Forward declaration from View
// This needs to be here to avoid a circular dependency; it's
// necessary to see if the view holder can be assignable to a CPU buffer
// for the purposes of type erasure
template <class T1, class T2>
struct is_always_assignable_impl;

}

namespace KokkosResilience {
namespace Impl {

template <class View>
struct unmanaged_view_type_like_impl;

template <template <class, class...> class ViewType, class DataType,
          class... Properties>
struct unmanaged_view_type_like_impl<ViewType<DataType, Properties...>> {
  using original_type = ViewType<DataType, Properties...>;
  using type =
      ViewType<typename original_type::traits::non_const_data_type,
               typename original_type::traits::array_layout, Kokkos::HostSpace,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using const_type =
      ViewType<typename original_type::traits::const_data_type,
               typename original_type::traits::array_layout, Kokkos::HostSpace,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
};

template <class ViewType>
using unmanaged_view_type_like =
    typename unmanaged_view_type_like_impl<ViewType>::type;

template <class ViewType>
using const_unmanaged_view_type_like =
    typename unmanaged_view_type_like_impl<ViewType>::const_type;

template <class ViewType, typename PtrType>
auto make_unmanaged_view_like(const ViewType &view, PtrType *buff) {
  using new_view_type = unmanaged_view_type_like<ViewType>;

  return new_view_type(
      reinterpret_cast<typename new_view_type::pointer_type>(buff),
      view.rank_dynamic > 0 ? view.extent(0) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      view.rank_dynamic > 1 ? view.extent(1) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      view.rank_dynamic > 2 ? view.extent(2) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      view.rank_dynamic > 3 ? view.extent(3) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      view.rank_dynamic > 4 ? view.extent(4) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      view.rank_dynamic > 5 ? view.extent(5) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      view.rank_dynamic > 6 ? view.extent(6) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      view.rank_dynamic > 7 ? view.extent(7) : KOKKOS_IMPL_CTOR_DEFAULT_ARG);
}

class ConstViewHolderImplBase {
 public:
  virtual ~ConstViewHolderImplBase() = default;

  size_t span() const { return m_span; }
  bool span_is_contiguous() const { return m_span_is_contiguous; }
  const void *data() const { return m_data; }
  std::string label() const { return m_label; }

  size_t data_type_size() const { return m_data_type_size; }
  bool is_host_space() const noexcept { return m_is_host_space; }

  virtual void deep_copy_to_buffer(unsigned char *buff) = 0;
  virtual ConstViewHolderImplBase *clone() const        = 0;

 protected:
  ConstViewHolderImplBase(std::size_t span, bool span_is_contiguous,
                          const void *data, std::string label,
                          std::size_t data_type_size, bool is_host_space)
      : m_span(span),
        m_span_is_contiguous(span_is_contiguous),
        m_data(data),
        m_label(std::move(label)),
        m_data_type_size(data_type_size),
        m_is_host_space(is_host_space) {}

 private:
  size_t m_span             = 0;
  bool m_span_is_contiguous = false;
  const void *m_data        = nullptr;
  std::string m_label;
  size_t m_data_type_size = 0;
  bool m_is_host_space    = false;
};

class ViewHolderImplBase {
 public:
  virtual ~ViewHolderImplBase() = default;

  size_t span() const { return m_span; }
  bool span_is_contiguous() const { return m_span_is_contiguous; }
  void *data() const { return m_data; }
  std::string label() const { return m_label; }

  size_t data_type_size() const { return m_data_type_size; }
  bool is_host_space() const noexcept { return m_is_host_space; }

  virtual void deep_copy_to_buffer(unsigned char *buff)   = 0;
  virtual void deep_copy_from_buffer(const unsigned char *buff) = 0;
  virtual ViewHolderImplBase *clone() const               = 0;

 protected:
  ViewHolderImplBase(std::size_t span, bool span_is_contiguous, void *data,
                     std::string label, std::size_t data_type_size,
                     bool is_host_space)
      : m_span(span),
        m_span_is_contiguous(span_is_contiguous),
        m_data(data),
        m_label(std::move(label)),
        m_data_type_size(data_type_size),
        m_is_host_space(is_host_space) {}

 private:
  size_t m_span             = 0;
  bool m_span_is_contiguous = false;
  void *m_data              = nullptr;
  std::string m_label;
  size_t m_data_type_size = 0;
  bool m_is_host_space    = false;
};

template <typename SrcViewType, typename DstViewType, typename Enabled = void>
struct ViewHolderImplDeepCopyImpl {
  static void copy_to_unmanaged(SrcViewType &, void *) {
    Kokkos::Impl::throw_runtime_exception(
        "Cannot deep copy a view holder to an incompatible view");
  }

  static void copy_from_unmanaged(DstViewType &, const void *) {
    Kokkos::Impl::throw_runtime_exception(
        "Cannot deep copy from a host unmanaged view holder to an incompatible "
        "view");
  }
};

template <typename SrcViewType, typename DstViewType>
struct ViewHolderImplDeepCopyImpl<SrcViewType, DstViewType,
                                  std::enable_if_t<Kokkos::is_always_assignable_impl<
                                      DstViewType, SrcViewType>::value>> {
  static void copy_to_unmanaged(SrcViewType &_src, void *_buff) {
    auto dst = make_unmanaged_view_like(
        _src, reinterpret_cast<unsigned char *>(_buff));
    deep_copy(dst, _src);
  }

  static void copy_from_unmanaged(DstViewType &_dst, const void *_buff) {
    auto src = Impl::make_unmanaged_view_like(
        _dst, reinterpret_cast<const unsigned char *>(_buff));
    deep_copy(_dst, src);
  }
};

template <typename View, typename Enable = void>
class ViewHolderImpl : public ViewHolderImplBase {
  static_assert(
      !std::is_same<typename View::traits::memory_space, Kokkos::AnonymousSpace>::value,
      "ViewHolder can't hold anonymous space view");

 public:
  virtual ~ViewHolderImpl() = default;

  explicit ViewHolderImpl(const View &view)
      : ViewHolderImplBase(
            view.span(), view.span_is_contiguous(), view.data(), view.label(),
            sizeof(typename View::value_type),
            std::is_same<typename View::memory_space, Kokkos::HostSpace>::value),
        m_view(view) {}

  void deep_copy_to_buffer(unsigned char *buff) override {
    using dst_type = unmanaged_view_type_like<View>;
    ViewHolderImplDeepCopyImpl<View, dst_type>::copy_to_unmanaged(m_view, buff);
  }

  void deep_copy_from_buffer(const unsigned char *buff) override {
    using src_type = const_unmanaged_view_type_like<View>;
    ViewHolderImplDeepCopyImpl<src_type, View>::copy_from_unmanaged(m_view,
                                                                    buff);
  }

  ViewHolderImpl *clone() const override { return new ViewHolderImpl(m_view); }

 private:
  View m_view;
};

template <class View>
class ViewHolderImpl<View, typename std::enable_if<std::is_const<
                               typename View::value_type>::value>::type>
    : public ConstViewHolderImplBase {
  static_assert(
      !std::is_same<typename View::traits::memory_space, Kokkos::AnonymousSpace>::value,
      "ViewHolder can't hold anonymous space view");

 public:
  virtual ~ViewHolderImpl() = default;

  explicit ViewHolderImpl(const View &view)
      : ConstViewHolderImplBase(
            view.span(), view.span_is_contiguous(), view.data(), view.label(),
            sizeof(typename View::value_type),
            std::is_same<typename View::memory_space, Kokkos::HostSpace>::value),
        m_view(view) {}

  void deep_copy_to_buffer(unsigned char *buff) override {
    using dst_type = unmanaged_view_type_like<View>;
    ViewHolderImplDeepCopyImpl<View, dst_type>::copy_to_unmanaged(m_view, buff);
  }

  ViewHolderImpl *clone() const override { return new ViewHolderImpl(m_view); }

 private:
  View m_view;
};
}  // namespace Impl

template <bool IsConst>
class BasicViewHolder {
 public:
  using value_type = std::conditional_t<IsConst, Impl::ConstViewHolderImplBase,
                                        Impl::ViewHolderImplBase>;

  BasicViewHolder() = default;

  BasicViewHolder(const BasicViewHolder &other)
      : m_impl(other.m_impl ? other.m_impl->clone() : nullptr) {}

  BasicViewHolder(BasicViewHolder &&other) { std::swap(m_impl, other.m_impl); }

  BasicViewHolder &operator=(const BasicViewHolder &other) {
    m_impl = std::unique_ptr<value_type>(other.m_impl ? other.m_impl->clone()
                                                      : nullptr);
    return *this;
  }

  BasicViewHolder &operator=(BasicViewHolder &&other) {
    std::swap(m_impl, other.m_impl);
    return *this;
  }

  std::conditional_t<IsConst, const void *, void *> data() const {
    return m_impl ? m_impl->data() : nullptr;
  }

  value_type *operator->() const noexcept { return m_impl.get(); }

 private:
  template <typename V>
  friend auto make_view_holder(const V &view);

  template <typename... Args>
  explicit BasicViewHolder(const Kokkos::View<Args...> &view)
      : m_impl(std::make_unique<Impl::ViewHolderImpl<Kokkos::View<Args...>>>(view)) {}

  std::unique_ptr<value_type> m_impl;
};

using ConstViewHolder = BasicViewHolder<true>;
using ViewHolder      = BasicViewHolder<false>;

template <typename V>
auto make_view_holder(const V &view) {
  using holder_type =
      typename std::conditional<std::is_const<typename V::value_type>::value,
                                ConstViewHolder, ViewHolder>::type;
  return holder_type(view);
}

}  // namespace KokkosResilience


#endif  // INC_RESILIENCE_VIEWHOLDER_HPP
