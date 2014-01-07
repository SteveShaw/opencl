/******************************************************************************\
 *           ___        __                                                    *
 *          /\_ \    __/\ \                                                   *
 *          \//\ \  /\_\ \ \____    ___   _____   _____      __               *
 *            \ \ \ \/\ \ \ '__`\  /'___\/\ '__`\/\ '__`\  /'__`\             *
 *             \_\ \_\ \ \ \ \L\ \/\ \__/\ \ \L\ \ \ \L\ \/\ \L\.\_           *
 *             /\____\\ \_\ \_,__/\ \____\\ \ ,__/\ \ ,__/\ \__/.\_\          *
 *             \/____/ \/_/\/___/  \/____/ \ \ \/  \ \ \/  \/__/\/_/          *
 *                                          \ \_\   \ \_\                     *
 *                                           \/_/    \/_/                     *
 *                                                                            *
 * Copyright (C) 2011-2013                                                    *
 * Dominik Charousset <dominik.charousset@haw-hamburg.de>                     *
 * Raphael Hiesgen <raphael.hiesgen@haw-hamburg.de>                           *
 *                                                                            *
 * This file is part of libcppa.                                              *
 * libcppa is free software: you can redistribute it and/or modify it under   *
 * the terms of the GNU Lesser General Public License as published by the     *
 * Free Software Foundation; either version 2.1 of the License,               *
 * or (at your option) any later version.                                     *
 *                                                                            *
 * libcppa is distributed in the hope that it will be useful,                 *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of             *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.                       *
 * See the GNU Lesser General Public License for more details.                *
 *                                                                            *
 * You should have received a copy of the GNU Lesser General Public License   *
 * along with libcppa. If not, see <http://www.gnu.org/licenses/>.            *
\******************************************************************************/

#ifndef CPPA_OPENCL_ACTOR_FACADE_HPP
#define CPPA_OPENCL_ACTOR_FACADE_HPP

#include <ostream>
#include <algorithm>
#include <stdexcept>
#include <iostream>

#include "cppa/cppa.hpp"

#include "cppa/channel.hpp"
#include "cppa/to_string.hpp"
#include "cppa/tuple_cast.hpp"
#include "cppa/intrusive_ptr.hpp"

#include "cppa/util/int_list.hpp"
#include "cppa/util/type_list.hpp"
#include "cppa/util/limited_vector.hpp"

#include "cppa/opencl/global.hpp"
#include "cppa/opencl/command.hpp"
#include "cppa/opencl/program.hpp"
#include "cppa/opencl/smart_ptr.hpp"

#include "cppa/detail/scheduled_actor_dummy.hpp"

namespace cppa {
namespace opencl {

class opencl_metainfo;

template <typename Signature>
class actor_facade;

template <typename Ret, typename... Args>
class actor_facade<Ret(Args...)> : public abstract_actor {

    friend class command<actor_facade, Ret>;

 public:

    typedef cow_tuple<typename util::rm_const_and_ref<Args>::type...>
            args_tuple;

    typedef std::function<optional<args_tuple>(any_tuple)> arg_mapping;
    typedef std::function<any_tuple(Ret&)> result_mapping;

    static intrusive_ptr<actor_facade>
    create(const program& prog, const char* kernel_name, arg_mapping map_args,
           result_mapping map_result, const dim_vec& global_dims,
           const dim_vec& offsets, const dim_vec& local_dims) {
        if (global_dims.empty()) {
            auto str = "OpenCL kernel needs at least 1 global dimension.";
            CPPA_LOGM_ERROR(detail::demangle(typeid(actor_facade)).c_str(),
                            str);
            throw std::runtime_error(str);
        }
        auto check_vec = [&](const dim_vec& vec, const char* name) {
            if (!vec.empty() && vec.size() != global_dims.size()) {
                std::ostringstream oss;
                oss << name << " vector is not empty, but "
                    << "its size differs from global dimensions vector's size";
                CPPA_LOGM_ERROR(detail::demangle<actor_facade>().c_str(),
                                oss.str());
                throw std::runtime_error(oss.str());
            }
        };
        check_vec(offsets, "offsets");
        check_vec(local_dims, "local dimensions");
        cl_int err{ 0 };
        kernel_ptr kernel;
        kernel.adopt(clCreateKernel(prog.m_program.get(), kernel_name, &err));
        if (err != CL_SUCCESS) {
            std::ostringstream oss;
            oss << "clCreateKernel: " << get_opencl_error(err);
            CPPA_LOGM_ERROR(detail::demangle<actor_facade>().c_str(),
                            oss.str());
            throw std::runtime_error(oss.str());
        }
        return new actor_facade<Ret(Args...)>{
            prog,       kernel,              global_dims,          offsets,
            local_dims, std::move(map_args), std::move(map_result)
        };
    }

    void enqueue(const message_header& hdr, any_tuple msg) override {
        CPPA_LOG_TRACE("");
        typename util::il_indices<util::type_list<Args...>>::type indices;
        enqueue_impl(hdr.sender, std::move(msg), hdr.id, indices);
    }

 private:

    actor_facade(const program& prog, kernel_ptr kernel,
                 const dim_vec& global_dimensions,
                 const dim_vec& global_offsets, const dim_vec& local_dimensions,
                 arg_mapping map_args, result_mapping map_result)
        : m_kernel(kernel), m_program(prog.m_program),
          m_context(prog.m_context), m_queue(prog.m_queue),
          m_global_dimensions(global_dimensions),
          m_global_offsets(global_offsets),
          m_local_dimensions(local_dimensions), m_map_args(std::move(map_args)),
          m_map_result(std::move(map_result)) {
        CPPA_LOG_TRACE("id: " << this->id());
    }

    template <long... Is>
    void enqueue_impl(const actor_ptr& sender, any_tuple msg, message_id id,
                      util::int_list<Is...>) {
        auto opt = m_map_args(std::move(msg));
        if (opt) {
            response_promise handle{ this, sender, id.response_id() };
            size_t ret_size = std::accumulate(m_global_dimensions.begin(),
                                              m_global_dimensions.end(), 1,
                                              std::multiplies<size_t>{});
            std::vector<mem_ptr> arguments;
            add_arguments_to_kernel<Ret>(arguments, ret_size,
                                         get_ref<Is>(*opt)...);
            auto cmd = make_counted<command<actor_facade, Ret>>(
                handle, this, std::move(arguments));
            cmd->enqueue();
        } else {
            CPPA_LOGMF(CPPA_ERROR, this,
                       "actor_facade::enqueue() tuple_cast failed.");
        }
    }

    typedef std::vector<mem_ptr> args_vec;

    kernel_ptr m_kernel;
    program_ptr m_program;
    context_ptr m_context;
    command_queue_ptr m_queue;
    dim_vec m_global_dimensions;
    dim_vec m_global_offsets;
    dim_vec m_local_dimensions;
    arg_mapping m_map_args;
    result_mapping m_map_result;

    void add_arguments_to_kernel_rec(args_vec& arguments) {
        cl_int err{ 0 };
        for (size_t i = 1; i < arguments.size(); ++i) {
            err = clSetKernelArg(m_kernel.get(), (i - 1), sizeof(cl_mem),
                                 static_cast<void*>(&arguments[i]));
            CPPA_LOG_ERROR_IF(err != CL_SUCCESS,
                              "clSetKernelArg: " << get_opencl_error(err));
        }
        err = clSetKernelArg(m_kernel.get(), arguments.size() - 1,
                             sizeof(cl_mem), static_cast<void*>(&arguments[0]));
        CPPA_LOG_ERROR_IF(err != CL_SUCCESS,
                          "clSetKernelArg: " << get_opencl_error(err));
    }

    template <typename T0, typename... Ts>
    void add_arguments_to_kernel_rec(args_vec& arguments, T0& arg0,
                                     Ts&... args) {
        cl_int err{ 0 };
        auto buf = clCreateBuffer(
            m_context.get(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(typename T0::value_type) * arg0.size(), arg0.data(), &err);
        if (err != CL_SUCCESS) {
            CPPA_LOGMF(CPPA_ERROR, this,
                       "clCreateBuffer: " << get_opencl_error(err));
        } else {
            mem_ptr tmp;
            tmp.adopt(std::move(buf));
            arguments.push_back(tmp);
            add_arguments_to_kernel_rec(arguments, args...);
        }
    }

    template <typename R, typename... Ts>
    void add_arguments_to_kernel(args_vec& arguments, size_t ret_size,
                                 Ts&&... args) {
        arguments.clear();
        cl_int err{ 0 };
        auto buf = clCreateBuffer(m_context.get(), CL_MEM_WRITE_ONLY,
                                  sizeof(typename R::value_type) * ret_size,
                                  nullptr, &err);
        if (err != CL_SUCCESS) {
            CPPA_LOGMF(CPPA_ERROR, this,
                       "clCreateBuffer: " << get_opencl_error(err));
        } else {
            mem_ptr tmp;
            tmp.adopt(std::move(buf));
            arguments.push_back(tmp);
            add_arguments_to_kernel_rec(arguments, std::forward<Ts>(args)...);
        }
    }

};

} // namespace opencl
} // namespace cppa

#endif // CPPA_OPENCL_ACTOR_FACADE_HPP
