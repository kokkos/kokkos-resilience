#ifndef RES_CUDA_SUBSCRIBER_HPP
#define RES_CUDA_SUBSCRIBER_HPP

#include <Kokkos_Macros.hpp>
#if defined(KR_ENABLE_CUDA)

#include <Kokkos_Core.hpp>
#include <iostream>
#include <cmath>
#include <map>
#include <typeinfo>
#include <unordered_map>
#include <sstream>

/*--------------------------------------------------------------------------
********************* ERROR MESSAGE GENERATION *****************************
 --------------------------------------------------------------------------*/

namespace KokkosResilience {

  struct CudaResilientSubscriber;

  // Generate usable error message

  static_assert(Kokkos::Experimental::is_hooks_policy<
                Kokkos::Experimental::SubscribableViewHooks<
 
                CudaResilientSubscriber> >::value, "Must be a hooks policy");

} // namespace KokkosResilience

/*----------------------------------------------------------------------------
 ******** STRUCT TO CHECK CORRECTNESS OF INDIVIDUAL ELEMENTS OF VIEWS ********
 ----------------------------------------------------------------------------*/

namespace KokkosResilience{

// Overseer to combine duplicates using checkers for accuracy
struct CombineDuplicatesBase
{
  // Virtual bool to return success flag
  virtual ~CombineDuplicatesBase() = default;
  virtual bool execute() = 0;
  // First index tracks first view, second index tracks second copy
  bool already_copied[2] = {false,false};

};

template< typename View >
struct CombineDuplicates: public CombineDuplicatesBase
{
  int duplicate_count = 0;
  View original; // Some template type parameter View not actually Views per se

  // Two copies allocated
  View copy[2]; // Two more of these template type parameters
  // will have to dereference optional<View> as pointer

  // ... hackz...
  //Kokkos::View <bool*> success{"success", 1};
  bool success = 0;

#if 0
  template <class Type>
inline __device__ std::enable_if_t< std::is_floating_point< Type>::value, bool> compare(
    const Type a, const Type b) 
   const{ return (abs(a - b) < 0.00000001);}
#endif
//KOKKOS_INLINE_FUNCTION void operator ()(int) const{}

KOKKOS_INLINE_FUNCTION static void combiner_kernel (int) {}

  // This is where combining usually happens, but it's always returning true for now
  bool execute() override
  {

    //success(0) = true;

    success = true;
    
    if (already_copied[0]==false || already_copied[1]==false) {
      Kokkos::abort("Aborted in CombineDuplicates, duplicate_count < 2");
    }
   // /*
    // Execute combine-duplicates operator in parallel
    else {
      std::cout << "@@@@@Before combine-duplicates parallel_for@@@@@ "<< std::endl;
 #if 0     
      auto err = cudaGetLastError();
      if (err != cudaSuccess)
      std::cout << "error:" << cudaGetErrorString(err); 
#endif
//Kokkos::Profiling::pushRegion("InCombiner"); 
      auto range_policy = Kokkos::RangePolicy<Kokkos::Cuda>(0,20);
      Kokkos::parallel_for("Combiner", range_policy, &combiner_kernel);
//Kokkos::Profiling::popRegion();
      std::cout << "After combine-duplicates parallel_for "<<std::endl;
      Kokkos::fence("FenceCombine");
   // */
    return success;
    // TODO: THERE IS A LOGIC PROBLEM HERE NOW, UNFIXED AND INTRODUCED BECAUSE OF KERNEL

  }
  //should never be hit, just to catch compiler errors  
return false;
}
//inline __device__ void operator ()(int) const{}
#if 0
  template<typename... Args>
  inline __device__ void operator ()(Args && ...its) const {
Kokkos::Profiling::pushRegion("InOperator");

    
    for (int j = 0; j < 2; j ++) {
      if(compare(copy[j](its...), original(its...))) {
        return;
      }
    }
    if (compare(copy[0](its...), copy[1](its...))){
      original(its...) = copy[0](its...); // just need any 2 to be same
      return;
    }
Kokkos::Profiling::popRegion();    
    // No match found, all three executions return a different number
    //success = false;

  }

#endif
};

}  // namespace KokkosResilience       

/*----------------------------------------------------------------------------
 **I********** SUBSCRIBER TO DUPLICATE VIEWS, COPY DATA ***********************
 ----------------------------------------------------------------------------*/

namespace KokkosResilience {

struct CudaResilientSubscriber {

  //ating for using subscriber only inside resilient parallel loops
  static int resilient_duplicate_counter; 
  
  // Creating map for duplicates: used for duplicate resolution per-kernel
  // Creating cache map of duplicates: used for tracking duplicates 
  // between kernels so that they are initialzied
  // only once. Re-initialize copies to be like original view only if 
  // original not in cache map
  
  using key_type = void *;  // key_type should be data() pointer


  // __device__ variable only allows globally allocated variables known at compile
  // Kokkos unordered map
  // data pointer, but not data of course, lives on the host  

  static std::unordered_map <key_type, CombineDuplicatesBase * > duplicates_map;
  static std::unordered_map <key_type, std::unique_ptr< CombineDuplicatesBase > > duplicates_cache;

  template<typename View>
  static CombineDuplicates<View> *
  get_duplicate_for( const View &original) {
   
    //Kokkos::Profiling::pushRegion("SubscriberInitialization"); 
    
    bool inserted = false;
    auto position = duplicates_cache.find(original.data());

    // True if got to end of cache and view wasn't found
    if (position == duplicates_cache.end()){
      // Insert view into cache map and flag
      inserted = true;
      position = duplicates_cache.try_emplace(original.data(),std::make_unique<CombineDuplicates<View>>()).first;
    }
   // Kokkos::Profiling::popRegion();
 //   std::cout << "Value of inserted = " << inserted << ", which reflects if view had to be inserted in cache map. Should see copies created with set_duplicate_view if 1." << std::endl;

    auto &res = *static_cast< CombineDuplicates<View> * >( position->second.get());

    // If inserted in the cache map then create copies and reinitialize
    if (inserted) {
      res.original = original; // The View type parameter becomes the original view
 
      // Reinitialize self to be like other (same dimensions, etc)
      for (int i = 0; i < 2; ++i) {
        
        set_duplicate_view(res.copy[i], original, i);
      }
    }
    return &res;
  }

  // Function which initializes the dimensions of the duplicating view
  template<typename View>
  KOKKOS_INLINE_FUNCTION
  static void set_duplicate_view(View &duplicate, const View &original, int duplicate_count) {
    std::stringstream label_ss;
    label_ss << original.label() << duplicate_count;
   
auto err = cudaGetLastError();
   if (err != cudaSuccess)
   std::cout << "error:" << cudaGetErrorString(err);

 //   std::cout << "New label from set_duplicate_view is " <<label_ss.str() <<std::endl;

     duplicate = View(Kokkos::view_alloc(label_ss.str(),Kokkos::WithoutInitializing), original.layout());
  }

  //Constant copy constructor for subscriber, test doing nothing but printing if the constructor is not empty.
  template<template<typename, typename ...> class V, typename T, typename... Args>
  static void copy_constructed( V < const T *, Args...> &self, const V < const T *, Args...> &other) {
  // If View is constant do nothing, not triggering the rest of the subscriber.
//  std::cout << "Constant copy constructor called. The value of IRPL is " << resilient_duplicate_counter << std::endl;


  }

  template< template< typename, typename ...> class V, typename T, typename... Args>
  static void copy_constructed( V < T *, Args... > &self, const V < T *, Args... > &other)
  {
     
    //std::cout << "Non-constant copy constructor called. The value of IRPL is " << resilient_duplicate_counter << std::endl;
 
    // If view is non-constant and in the parallel loop, cascade the rest of the subscriber
    if (resilient_duplicate_counter > 0) {
      auto err = cudaGetLastError();
      if (err != cudaSuccess)
        std::cout << "error:" << cudaGetErrorString(err);     

  //    std::cout << "This statement is right before get_duplicate_for, which is triggered every time, but the map-emplacement fails if the entry already exists. I should really fix that comment." << std::endl;
      // This won't be triggered if the entry already exists
      auto *combiner = get_duplicate_for(other);
      auto res = duplicates_map.try_emplace(other.data(),combiner);
      auto &c = dynamic_cast< CombineDuplicates< V<T*, Args...> > & > (*res.first->second);
  
      // If it is the first copy constructor
      // The first copy constructor in a parallel_for for the given view
      if (res.second) {
       assert(resilient_duplicate_counter == 1);
 //       std::cout << "This statement in if loop setting c.duplicate_count = 0. c.duplicate_count is " << c.duplicate_count << std::endl;
      }

      c.duplicate_count = resilient_duplicate_counter-1;
     
      // TODO: Checked logic, access point definitely correct
      if (!c.already_copied[c.duplicate_count]){ 
      self = c.copy[c.duplicate_count];
//      std::cout << "*****In copy constructor, self has now been set to copy " << c.duplicate_count << "****" << std::endl;      
      c.already_copied[c.duplicate_count]=true;
     // Copy all data, every time
      Kokkos::deep_copy(self, other);
      
//      std::cout << std::endl << "This is the end of the constructor after Kokkos::deep_copy." << std::endl <<std::endl;
}//alread_copie
    }
  }

  //Subscriber format requires these templates as well
  template <typename View>
  static void move_constructed(View &self, const View &other) {}

  template <typename View>
  static void move_assigned(View &self, const View &other) {}
  template <typename View>
  static void copy_assigned(View &self, const View &other) {}

}; // end CudaResilientSubscriber

inline void clear_duplicates_map() {
  for(auto&&[key,duplicate]:KokkosResilience::CudaResilientSubscriber::duplicates_map){
    duplicate->already_copied[0]=false;
    duplicate->already_copied[1]=false;
  }
  
  KokkosResilience::CudaResilientSubscriber::duplicates_map.clear();
}

//kokkos inline marks function as host and device, appears on both
inline void clear_duplicates_cache() {
  KokkosResilience::CudaResilientSubscriber::duplicates_cache.clear();
}

} //namespace KokkosResilience


#endif //defined(KR_ENABLE_CUDA)
#endif //RES_CUDA_SUBSCRIBER_HPP

