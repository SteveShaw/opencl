/******************************************************************************
 *                       ____    _    _____                                   *
 *                      / ___|  / \  |  ___|    C++                           *
 *                     | |     / _ \ | |_       Actor                         *
 *                     | |___ / ___ \|  _|      Framework                     *
 *                      \____/_/   \_|_|                                      *
 *                                                                            *
 * Copyright (C) 2011 - 2015                                                  *
 * Dominik Charousset <dominik.charousset (at) haw-hamburg.de>                *
 *                                                                            *
 * Distributed under the terms and conditions of the BSD 3-Clause License or  *
 * (at your option) under the terms and conditions of the Boost Software      *
 * License 1.0. See accompanying files LICENSE and LICENSE_ALTERNATIVE.       *
 *                                                                            *
 * If you did not receive a copy of the license files, see                    *
 * http://opensource.org/licenses/BSD-3-Clause and                            *
 * http://www.boost.org/LICENSE_1_0.txt.                                      *
 ******************************************************************************/

#ifndef CAF_OPENCL_DEVICE_HPP
#define CAF_OPENCL_DEVICE_HPP

#include <vector>

#include "caf/opencl/global.hpp"
#include "caf/opencl/smart_ptr.hpp"

namespace caf {
namespace opencl {

class program;

class device {
  friend class program;

public:
  /// Intialize a new device in a context using a sepcific device_id
  static device create(context_ptr context, device_ptr device_id, unsigned id);
  /// Get the id assigned by caf
  inline unsigned get_id();
  /// Returns device info on CL_DEVICE_ADDRESS_BITS
  inline cl_uint get_address_bits();
  /// Returns device info on CL_DEVICE_ENDIAN_LITTLE
  inline cl_bool get_little_endian();
  /// Returns device info on CL_DEVICE_GLOBAL_MEM_CACHE_SIZE
  inline cl_ulong get_global_mem_cache_size();
  /// Returns device info on CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE
  inline cl_uint get_global_mem_cacheline_size();
  /// Returns device info on CL_DEVICE_GLOBAL_MEM_SIZE
  inline cl_ulong get_global_mem_size();
  /// Returns device info on CL_DEVICE_HOST_UNIFIED_MEMORY
  inline cl_bool get_host_unified_memory();
  /// Returns device info on CL_DEVICE_LOCAL_MEM_SIZE
  inline cl_ulong get_local_mem_size();
  /// Returns device info on CL_DEVICE_LOCAL_MEM_TYPE
  inline cl_uint get_local_mem_type();
  /// Returns device info on CL_DEVICE_MAX_CLOCK_FREQUENCY
  inline cl_uint get_max_clock_frequency();
  /// Returns device info on CL_DEVICE_MAX_COMPUTE_UNITS
  inline cl_uint get_max_compute_units();
  /// Returns device info on CL_DEVICE_MAX_CONSTANT_ARGS
  inline cl_uint get_max_constant_args();
  /// Returns device info on CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE
  inline cl_ulong get_max_constant_buffer_size();
  /// Returns device info on CL_DEVICE_MAX_MEM_ALLOC_SIZE
  inline cl_ulong get_max_mem_alloc_size();
  /// Returns device info on CL_DEVICE_MAX_PARAMETER_SIZE
  inline size_t get_max_parameter_size();
  /// Returns device info on CL_DEVICE_MAX_WORK_GROUP_SIZE
  inline size_t get_max_work_group_size();
  /// Returns device info on CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS
  inline cl_uint get_max_work_item_dimensions();
  /// Returns device info on CL_DEVICE_PROFILING_TIMER_RESOLUTION
  inline size_t get_profiling_timer_resolution();
  /// Returns device info on CL_DEVICE_MAX_WORK_ITEM_SIZES
  inline const dim_vec& get_max_work_item_sizes();
  /// Returns device info on CL_DEVICE_TYPE
  inline device_type get_device_type();
  /// Returns device info on CL_DEVICE_EXTENSIONS
  inline const std::vector<std::string>& get_extensions();
  /// Returns device info on CL_DEVICE_OPENCL_C_VERSION
  inline const std::string get_opencl_c_version();
  /// Returns device info on CL_DEVICE_VENDOR
  inline const std::string get_device_vendor();
  /// Returns device info on CL_DEVICE_VERSION
  inline const std::string get_device_version();
  /// Returns device info on CL_DRIVER_VERSION
  inline const std::string get_driver_version();
  /// Returns device info on CL_DEVICE_NAME
  inline const std::string& get_name();

private:
  device(device_ptr device_id, command_queue_ptr queue, context_ptr context,
         unsigned id);
  template <class T>
  static T info(device_ptr device_id, unsigned info_flag);
  static std::string info_string(device_ptr device_id, unsigned info_flag);
  device_ptr device_id_;
  command_queue_ptr command_queue_;
  context_ptr context_;
  unsigned id_;

  bool profiling_enabled_;              // CL_DEVICE_QUEUE_PROPERTIES
  bool out_of_order_execution_;         // CL_DEVICE_QUEUE_PROPERTIES

  cl_uint address_bits_;                // CL_DEVICE_ADDRESS_BITS
  cl_bool little_endian_;               // CL_DEVICE_ENDIAN_LITTLE
  cl_ulong global_mem_cache_size_;      // CL_DEVICE_GLOBAL_MEM_CACHE_SIZE
  cl_uint global_mem_cacheline_size_;   // CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE
  cl_ulong global_mem_size_;            // CL_DEVICE_GLOBAL_MEM_SIZE
  cl_bool host_unified_memory_;         // CL_DEVICE_HOST_UNIFIED_MEMORY
  cl_ulong local_mem_size_;             // CL_DEVICE_LOCAL_MEM_SIZE
  cl_uint local_mem_type_;              // CL_DEVICE_LOCAL_MEM_TYPE
  cl_uint max_clock_frequency_;         // CL_DEVICE_MAX_CLOCK_FREQUENCY
  cl_uint max_compute_units_;           // CL_DEVICE_MAX_COMPUTE_UNITS
  cl_uint max_constant_args_;           // CL_DEVICE_MAX_CONSTANT_ARGS
  cl_ulong max_constant_buffer_size_;   // CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE
  cl_ulong max_mem_alloc_size_;         // CL_DEVICE_MAX_MEM_ALLOC_SIZE
  size_t max_parameter_size_;           // CL_DEVICE_MAX_PARAMETER_SIZE
  size_t max_work_group_size_;          // CL_DEVICE_MAX_WORK_GROUP_SIZE
  cl_uint max_work_item_dimensions_;    // CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS
  size_t profiling_timer_resolution_;   // CL_DEVICE_PROFILING_TIMER_RESOLUTION
  dim_vec max_work_item_sizes_;         // CL_DEVICE_MAX_WORK_ITEM_SIZES
  device_type device_type_;             // CL_DEVICE_TYPE
  std::vector<std::string> extensions_; // CL_DEVICE_EXTENSIONS
  std::string opencl_c_version_;        // CL_DEVICE_OPENCL_C_VERSION
  std::string device_vendor_;           // CL_DEVICE_VENDOR
  std::string device_version_;          // CL_DEVICE_VERSION
  std::string driver_version_;          // CL_DRIVER_VERSION
  std::string name_;                    // CL_DEVICE_NAME
};

/******************************************************************************\
 *                 implementation of inline member functions                  *
\******************************************************************************/

inline unsigned device::get_id() {
  return id_;
}

inline cl_uint device::get_address_bits() {
  return address_bits_;
}

inline cl_bool device::get_little_endian() {
  return little_endian_;
}

inline cl_ulong device::get_global_mem_cache_size() {
  return global_mem_cache_size_;
}

inline cl_uint device::get_global_mem_cacheline_size() {
  return global_mem_cacheline_size_;
}

inline cl_ulong device::get_global_mem_size() {
  return global_mem_size_;
}

inline cl_bool device::get_host_unified_memory() {
  return host_unified_memory_;
}

inline cl_ulong device::get_local_mem_size() {
  return local_mem_size_;
}

inline cl_uint device::get_local_mem_type() {
  return local_mem_size_;
}

inline cl_uint device::get_max_clock_frequency() {
  return max_clock_frequency_;
}

inline cl_uint device::get_max_compute_units() {
  return max_compute_units_;
}

inline cl_uint device::get_max_constant_args() {
  return max_constant_args_;
}

inline cl_ulong device::get_max_constant_buffer_size() {
  return max_constant_buffer_size_;
}

inline cl_ulong device::get_max_mem_alloc_size() {
  return max_mem_alloc_size_;
}

inline size_t device::get_max_parameter_size() {
  return max_parameter_size_;
}

inline size_t device::get_max_work_group_size() {
  return max_work_group_size_;
}

inline cl_uint device::get_max_work_item_dimensions() {
  return max_work_item_dimensions_;
}

inline size_t device::get_profiling_timer_resolution() {
  return profiling_timer_resolution_;
}

inline const dim_vec& device::get_max_work_item_sizes() {
  return max_work_item_sizes_;
}

inline device_type device::get_device_type() {
  return device_type_;
}

inline const std::vector<std::string>& device::get_extensions() {
  return extensions_;
}

inline const std::string device::get_opencl_c_version() {
  return opencl_c_version_;
}

inline const std::string device::get_device_vendor() {
  return device_vendor_;
}

inline const std::string device::get_device_version() {
  return device_version_;
}

inline const std::string device::get_driver_version() {
  return driver_version_;
}

inline const std::string& device::get_name() {
  return name_;
}

} // namespace opencl
} // namespace caf

#endif // CAF_OPENCL_DEVICE_HPP
