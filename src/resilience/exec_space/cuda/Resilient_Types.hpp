#include "Resilient_CudaSpace.hpp"
#include "Resilient_Cuda.hpp"
#include "Resilient_Cuda_Subscriber.hpp"
#include <Kokkos_Core.hpp>

namespace KokkosResilience{

 struct ResilientDuplicatesSubscriber;

 using ResilientView = Kokkos::View< double*, Kokkos::LayoutLeft, KokkosResilience::ResCudaSpace,
                      Kokkos::Experimental::SubscribableViewHooks<
                              KokkosResilience::ResilientDuplicatesSubscriber>>;

}
