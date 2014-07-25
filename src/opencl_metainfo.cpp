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
 * License 1.0. See accompanying files LICENSE and LICENCE_ALTERNATIVE.       *
 *                                                                            *
 * If you did not receive a copy of the license files, see                    *
 * http://opensource.org/licenses/BSD-3-Clause and                            *
 * http://www.boost.org/LICENSE_1_0.txt.                                      *
 ******************************************************************************/

#include "cppa/opencl/opencl_metainfo.hpp"

using namespace std;

namespace caf {
namespace opencl {

const std::vector<device_info> opencl_metainfo::get_devices() const {
    return m_devices;
}

void opencl_metainfo::initialize()
{
    cl_int err{0};


    // get number of available platforms
    cl_uint number_of_platforms;
    err = clGetPlatformIDs(0, nullptr, &number_of_platforms);
    if (err != CL_SUCCESS) {
        ostringstream oss;
        oss << "clGetPlatformIDs (getting number of platforms): "
            << get_opencl_error(err);
        CPPA_LOGMF(CPPA_ERROR, oss.str());
        throw logic_error(oss.str());
    }


    // get platform ids
    vector<cl_platform_id> ids(number_of_platforms);
    err = clGetPlatformIDs(ids.size(), ids.data(), nullptr);
    if (err != CL_SUCCESS) {
        ostringstream oss;
        oss << "clGetPlatformIDs (getting platform ids): "
            << get_opencl_error(err);
        CPPA_LOGMF(CPPA_ERROR, oss.str());
        throw logic_error(oss.str());
    }


    // find gpu devices on our platform
    int pid{0};
    cl_uint num_devices{0};
    cl_device_type dev_type{CL_DEVICE_TYPE_GPU};
    err = clGetDeviceIDs(ids[pid], dev_type, 0, nullptr, &num_devices);
    if (err == CL_DEVICE_NOT_FOUND) {
        CPPA_LOG_TRACE("No gpu devices found. Looking for cpu devices.");
        cout << "No gpu devices found. Looking for cpu devices." << endl;
        dev_type = CL_DEVICE_TYPE_CPU;
        err = clGetDeviceIDs(ids[pid], dev_type, 0, nullptr, &num_devices);
    }
    if (err != CL_SUCCESS) {
        ostringstream oss;
        oss << "clGetDeviceIDs: " << get_opencl_error(err);
        CPPA_LOGMF(CPPA_ERROR, oss.str());
        throw runtime_error(oss.str());
    }
    vector<cl_device_id> devices(num_devices);
    err = clGetDeviceIDs(ids[pid], dev_type, num_devices, devices.data(), nullptr);
    if (err != CL_SUCCESS) {
        ostringstream oss;
        oss << "clGetDeviceIDs: " << get_opencl_error(err);
        CPPA_LOGMF(CPPA_ERROR, oss.str());
        throw runtime_error(oss.str());
    }

    auto pfn_notify = [](const char *errinfo,
                         const void *,
                         size_t,
                         void *) {
        CPPA_LOGC_ERROR("cppa::opencl::opencl_metainfo",
                        "initialize",
                        "\n##### Error message via pfn_notify #####\n" +
                        string(errinfo) +
                        "\n########################################");
    };

    // create a context
    m_context.adopt(clCreateContext(0,
                                    devices.size(),
                                    devices.data(),
                                    pfn_notify,
                                    nullptr,
                                    &err));
    if (err != CL_SUCCESS) {
        ostringstream oss;
        oss << "clCreateContext: " << get_opencl_error(err);
        CPPA_LOGMF(CPPA_ERROR, oss.str());
        throw runtime_error(oss.str());
    }


    for (auto& d : devices) {
        CPPA_LOG_TRACE("Creating command queue for device(s).");
        device_ptr device;
        device.adopt(d);
        size_t return_size{0};
        static constexpr size_t buf_size = 128;
        char buf[buf_size];
        err = clGetDeviceInfo(device.get(), CL_DEVICE_NAME, buf_size, buf, &return_size);
        if (err != CL_SUCCESS) {
            CPPA_LOGMF(CPPA_ERROR, "clGetDeviceInfo (CL_DEVICE_NAME): "
                                   << get_opencl_error(err));
            fill(buf, buf+buf_size, 0);
        }
        command_queue_ptr cmd_queue;
        cmd_queue.adopt(clCreateCommandQueue(m_context.get(),
                                             device.get(),
                                             CL_QUEUE_PROFILING_ENABLE,
                                             &err));
        if (err != CL_SUCCESS) {
            CPPA_LOGMF(CPPA_DEBUG, "Could not create command queue for device "
                                   << buf << ": " << get_opencl_error(err));
        }
        else {
            size_t max_work_group_size{0};
            err = clGetDeviceInfo(device.get(),
                                  CL_DEVICE_MAX_WORK_GROUP_SIZE,
                                  sizeof(size_t),
                                  &max_work_group_size,
                                  &return_size);
            if (err != CL_SUCCESS) {
                ostringstream oss;
                oss << "clGetDeviceInfo ("
                    << "CL_DEVICE_MAX_WORK_GROUP_SIZE): "
                    << get_opencl_error(err);
                CPPA_LOGMF(CPPA_ERROR, oss.str());
                throw runtime_error(oss.str());
            }
            cl_uint max_work_item_dimensions = 0;
            err = clGetDeviceInfo(device.get(),
                                  CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                                  sizeof(cl_uint),
                                  &max_work_item_dimensions,
                                  &return_size);
            if (err != CL_SUCCESS) {
                ostringstream oss;
                oss << "clGetDeviceInfo ("
                    << "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS): "
                    << get_opencl_error(err);
                CPPA_LOGMF(CPPA_ERROR, oss.str());
                throw runtime_error(oss.str());
            }
            dim_vec max_work_items_per_dim(max_work_item_dimensions);
            err = clGetDeviceInfo(device.get(),
                                  CL_DEVICE_MAX_WORK_ITEM_SIZES,
                                  sizeof(size_t)*max_work_item_dimensions,
                                  max_work_items_per_dim.data(),
                                  &return_size);
            if (err != CL_SUCCESS) {
                ostringstream oss;
                oss << "clGetDeviceInfo ("
                    << "CL_DEVICE_MAX_WORK_ITEM_SIZES): "
                    << get_opencl_error(err);
                CPPA_LOGMF(CPPA_ERROR, oss.str());
                throw runtime_error(oss.str());
            }
            device_info dev_info{device,
                                 cmd_queue,
                                 max_work_group_size,
                                 max_work_item_dimensions,
                                 max_work_items_per_dim};
            m_devices.push_back(move(dev_info));
        }
    }

    if (m_devices.empty()) {
        ostringstream oss;
        oss << "Could not create a command queue for "
            << "any present device.";
        CPPA_LOGMF(CPPA_ERROR, oss.str());
        throw runtime_error(oss.str());
    }
}

void opencl_metainfo::destroy() {
    delete this;
}

void opencl_metainfo::dispose() {
    delete this;
}

opencl_metainfo* get_opencl_metainfo() {
    return detail::singleton_manager::get_opencl_metainfo();
}

} // namespace opencl
} // namespace caf


