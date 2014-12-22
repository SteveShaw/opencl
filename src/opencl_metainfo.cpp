/******************************************************************************
 *                       ____    _    _____                                   *
 *                      / ___|  / \  |  ___|    C++                           *
 *                     | |     / _ \ | |_       Actor                         *
 *                     | |___ / ___ \|  _|      Framework                     *
 *                      \____/_/   \_|_|                                      *
 *                                                                            *
 * Copyright (C) 2011 - 2014                                                  *
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

#include "caf/opencl/opencl_metainfo.hpp"

#define CAF_CLF(funname) #funname , funname

namespace caf {
namespace opencl {

namespace {

void throwcl(const char* fname, cl_int err) {
  if (err != CL_SUCCESS) {
    std::string errstr = fname;
    errstr += ": ";
    errstr += get_opencl_error(err);
    throw std::runtime_error(std::move(errstr));
  }
}

// call convention for simply calling a function, not using the last argument
template <class F, class... Ts>
void callcl(const char* fname, F f, Ts&&... vs) {
  throwcl(fname, f(std::forward<Ts>(vs)..., nullptr));
}

// call convention with `result` argument at the end returning `err`, not
// using the second last argument (set to nullptr) nor the one before (set to 0)
template <class R, class F, class... Ts>
R v1get(const char* fname, F f, Ts&&... vs) {
  R result;
  throwcl(fname, f(std::forward<Ts>(vs)..., cl_uint{0}, nullptr, &result));
  return result;
}

// call convention with `err` argument at the end returning `result`
template <class F, class... Ts>
auto v2get(const char* fname, F f, Ts&&... vs)
-> decltype(f(std::forward<Ts>(vs)..., nullptr)) {
  cl_int err;
  auto result = f(std::forward<Ts>(vs)..., &err);
  throwcl(fname, err);
  return result;
}

// call convention with `result` argument at second last position (preceeded by
// its size) followed by an ingored void* argument (nullptr) returning `err`
template <class R, class F, class... Ts>
R v3get(const char* fname, F f, Ts&&... vs) {
  R result;
  throwcl(fname, f(std::forward<Ts>(vs)..., sizeof(R), &result, nullptr));
  return result;
}

void pfn_notify(const char* errinfo, const void*, size_t, void*) {
  CAF_LOGC_ERROR("caf::opencl::opencl_metainfo", "initialize",
                 "\n##### Error message via pfn_notify #####\n"
                 << errinfo <<
                 "\n########################################");
}

} // namespace <anonymous>

opencl_metainfo* opencl_metainfo::instance() {
  auto sid = detail::singletons::opencl_plugin_id;
  auto fac = [] { return new opencl_metainfo; };
  auto res = detail::singletons::get_plugin_singleton(sid, fac);
  return static_cast<opencl_metainfo*>(res);
}

const std::vector<device_info> opencl_metainfo::get_devices() const {
  return m_devices;
}

void opencl_metainfo::initialize() {
  // get number of available platforms
  auto num_platforms = v1get<cl_uint>(CAF_CLF(clGetPlatformIDs));
  // get platform ids
  std::vector<cl_platform_id> platforms(num_platforms);
  callcl(CAF_CLF(clGetPlatformIDs), num_platforms, platforms.data());
  if (platforms.empty()) {
    throw std::runtime_error("no OpenCL platform found");
  }
  // support multiple platforms -> "for (auto platform : platforms)"?
  auto platform = platforms.front();
  // detect how many devices we got
  cl_uint num_devs = 0;
  cl_device_type dev_type = CL_DEVICE_TYPE_GPU;
  // try get some GPU devices and try falling back to CPU devices on error
  try {
    num_devs = v1get<cl_uint>(CAF_CLF(clGetDeviceIDs), platform, dev_type);
  }
  catch (std::runtime_error&) {
    dev_type = CL_DEVICE_TYPE_CPU;
    num_devs = v1get<cl_uint>(CAF_CLF(clGetDeviceIDs), platform, dev_type);
  }
  // get available devices
  std::vector<cl_device_id> ds(num_devs);
  callcl(CAF_CLF(clGetDeviceIDs), platform, dev_type, num_devs, ds.data());
  std::vector<device_ptr> devices(num_devs);
  // lift raw pointer as returned by OpenCL to C++ smart pointers
  auto lift = [](cl_device_id ptr) { return device_ptr{ptr, false}; };
  std::transform(ds.begin(), ds.end(), devices.begin(), lift);
  // create a context
  m_context.adopt(v2get(CAF_CLF(clCreateContext), nullptr, num_devs, ds.data(),
                        pfn_notify, nullptr));
  for (auto& device : devices) {
    CAF_LOG_DEBUG("creating command queue for device(s)");
    command_queue_ptr cmd_queue;
    try {
      cmd_queue.adopt(v2get(CAF_CLF(clCreateCommandQueue), m_context.get(),
                            device.get(), CL_QUEUE_PROFILING_ENABLE));
    }
    catch (std::runtime_error&) {
      CAF_LOG_DEBUG("unable to create command queue for device");
    }
    if (cmd_queue) {
      auto max_wgs = v3get<size_t>(CAF_CLF(clGetDeviceInfo), device.get(),
                                   CL_DEVICE_MAX_WORK_GROUP_SIZE);
      auto max_wid = v3get<cl_uint>(CAF_CLF(clGetDeviceInfo), device.get(),
                                    CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS);
      dim_vec max_wi_per_dim(max_wid);
      callcl(CAF_CLF(clGetDeviceInfo), device.get(),
             CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t) * max_wid,
             max_wi_per_dim.data());
      m_devices.push_back(device_info{std::move(device), std::move(cmd_queue),
                                      max_wgs, max_wid, max_wi_per_dim});
    }
  }
  if (m_devices.empty()) {
    std::string errstr = "could not create a command queue for any device";
    CAF_LOG_ERROR(errstr);
    throw std::runtime_error(std::move(errstr));
  }
}

void opencl_metainfo::dispose() {
  delete this;
}

void opencl_metainfo::stop() {
  // nop
}

} // namespace opencl
} // namespace caf
