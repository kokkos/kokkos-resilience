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
#ifndef INC_RESILIENCE_OPENMP_ERROR_INJECTOR_HPP
#define INC_RESILIENCE_OPENMP_ERROR_INJECTOR_HPP

#include <Kokkos_Macros.hpp>
#if defined(KOKKOS_ENABLE_OPENMP)

#include "Resilient_OpenMP_Subscriber.hpp"

namespace KokkosResilience{

// Struct to gate error insertion
struct Error{
  explicit Error(double rate) : error_rate(rate), geometric(rate){}

  double error_rate;
  std::geometric_distribution<> geometric{error_rate};
};

inline std::optional<Error> global_error_settings;

struct ErrorInjectionTracking{
  inline static size_t error_counter;
  inline static std::mt19937 random_gen{0};
  inline static size_t global_next_inject = 0;
  inline static std::chrono::duration<long int, std::nano> elapsed_seconds{};
  inline static std::chrono::duration<long int, std::nano> total_error_time{};
};

// Calculates coordinate formulas from linear iterator
template< typename View>
auto get_inject_indices_array( const View &view, std::size_t next_inject ){

  std::array<std::size_t, 8> indices {};
  size_t next_inject_copy = next_inject;

  // View.extent() returns 1 for uninitialized dimensions
  // this array returns accurate coordinates up to the existing view rank 
  // and zero for the rest, which are truncated by view.access() in the main injector
  // assumes column-major (Fortran) ordering
  for(int i=0;i<8;i++){
    indices[i] = next_inject_copy % view.extent(i);
    next_inject_copy /= view.extent(i);
  }

  auto tuple_indices = std::tuple_cat(indices);
  return tuple_indices;
}

template <typename View>	
void error_injection(View& original, View& copy_0, View& copy_1)
{
#ifdef KR_TRIPLE_MODULAR_REDUNDANCY
  //Any-dimensional TMR error injector
  size_t total_extent = original.size();

  //requires error in range, unless view size too small
  if (total_extent !=1 && (ErrorInjectionTracking::global_next_inject > total_extent))
  {
    while (ErrorInjectionTracking::global_next_inject>total_extent){
      ErrorInjectionTracking::global_next_inject = ErrorInjectionTracking::global_next_inject - total_extent;
    }
  }

  auto access = [](auto *view, auto... idcs) -> decltype (auto)
                {return view->access(idcs...);};
  size_t next_inject = ErrorInjectionTracking::global_next_inject;

  for (int j = 0; j<=2; j++){
    while (next_inject < total_extent)
    {
      auto indices = get_inject_indices_array( original, next_inject );
      if (j==0){//Inject in the original if j is 0
	auto view_tuple = std::tuple_cat(std::make_tuple(&original), indices);
        //replace value with noise
	std::apply(access, view_tuple)
  	              = static_cast<typename View::value_type>(ErrorInjectionTracking::random_gen());
        ErrorInjectionTracking::error_counter++;
      }
      else if(j==1){//Else inject in one of the other two copies, copy[0]
	auto view_tuple = std::tuple_cat(std::make_tuple(&copy_0), indices);
	std::apply(access, view_tuple)
    	            = static_cast<typename View::value_type>(ErrorInjectionTracking::random_gen());
        ErrorInjectionTracking::error_counter++;
      }
      else{//or copy[1]
	auto view_tuple = std::tuple_cat(std::make_tuple(&copy_1), indices);
	std::apply(access, view_tuple)
    	            = static_cast<typename View::value_type>(ErrorInjectionTracking::random_gen());
        ErrorInjectionTracking::error_counter++;
      }
      next_inject = global_error_settings->geometric(ErrorInjectionTracking::random_gen)+next_inject+1;
    }
    if(total_extent != 1){
      next_inject = next_inject - total_extent;
    }
  }
#endif
}// end inject_error

KOKKOS_INLINE_FUNCTION
void print_total_error_time() {

  static std::mutex global_time_mutex;	
  global_time_mutex.lock();
  std::cout << "The value of ErrorInjectionTracking::total_error_time.count() is " << ErrorInjectionTracking::total_error_time.count() << " nanoseconds." << std::endl;
  std::cout << "The total number of errors inserted is " << ErrorInjectionTracking::error_counter << " errors." << std::endl;
  global_time_mutex.unlock();

}

}//KokkosResilience

#endif //defined(KOKKOS_ENABLE_OPENMP)
#endif //INC_RESILIENCE_OPENMP_ERROR_INJECTOR_HPP




