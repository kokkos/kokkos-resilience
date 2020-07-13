#ifndef INC_RESILIENCE_CHECKPOINTDATA_HPP
#define INC_RESILIENCE_CHECKPOINTDATA_HPP

#include <hpx/modules/checkpoint_base.hpp>
#include <hpx/modules/serialization.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_ViewHooks.hpp>

#include <cstddef>
#include <string>
#include <tuple>
#include <utility>

namespace KokkosResilience {

namespace Detail {

// CheckpointData: helper enabling to checkpoint arbitrary data using HPX check-
// pointing

template <typename... Ts>
class CheckpointData : public Kokkos::ViewHolderBase {
 public:
  CheckpointData(std::string const& label, Ts const&... ts)
      : label_(label), checkpoint_data_() {
    hpx::util::save_checkpoint_data(checkpoint_data_, ts...);
  }

  std::size_t span() const override { return checkpoint_data_.size(); }

  bool span_is_contiguous() const override { return true; }

  void const* data() const override { return checkpoint_data_.data(); }
  void* data() override { return checkpoint_data_.data(); }

  CheckpointData* clone() const override { return new CheckpointData(*this); }

  std::string label() const noexcept override { return label_; }
  std::size_t data_type_size() const noexcept override { return sizeof(char); }

  bool is_hostspace() const noexcept override { return true; }

  void deep_copy_to_buffer(unsigned char* buff) override {
    // TODO: implement
  }

  void deep_copy_from_buffer(unsigned char* buff) override {
    // TODO: implement
  }

 private:
  std::string label_;
  std::vector<char> checkpoint_data_;
};

template <typename... Ts>
CheckpointData(std::string const&, Ts const&...) -> CheckpointData<Ts...>;

// RestoreData: helper enabling to restore arbitrary data using HPX check-
// pointing

template <typename... Ts>
class RestoreData : public Kokkos::ViewHolderBase {
 public:
  RestoreData(std::string const& label, Ts&... ts)
      : label_(label),
        data_(std::forward_as_tuple(ts...)),
        checkpoint_data_(hpx::util::prepare_checkpoint_data(ts...)) {}

  ~RestoreData() {
    restore_checkpoint(std::make_index_sequence<sizeof...(Ts)>{});
  }

  std::size_t span() const override { return checkpoint_data_.size(); }

  bool span_is_contiguous() const override { return true; }

  void const* data() const override { return checkpoint_data_.data(); }
  void* data() override { return checkpoint_data_.data(); }

  RestoreData* clone() const override { return new RestoreData(*this); }

  std::string label() const noexcept override { return label_; }
  std::size_t data_type_size() const noexcept override { return sizeof(char); }

  bool is_hostspace() const noexcept override { return true; }

  void deep_copy_to_buffer(unsigned char* buff) override {
    // TODO: implement
  }

  void deep_copy_from_buffer(unsigned char* buff) override {
    // TODO: implement
  }

 private:
  template <std::size_t... Is>
  void restore_checkpoint(std::index_sequence<Is...>) {
    hpx::util::restore_checkpoint_data(checkpoint_data_, std::get<Is>(data_)...);
  }

 private:
  std::string label_;
  std::tuple<Ts&...> data_;
  std::vector<char> checkpoint_data_;
};

template <typename... Ts>
RestoreData(std::string const&, Ts&...) -> RestoreData<Ts...>;

}  // namespace Detail
}  // namespace KokkosResilience

#endif
