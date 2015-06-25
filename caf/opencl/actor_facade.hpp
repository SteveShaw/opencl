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

#ifndef CAF_OPENCL_ACTOR_FACADE_HPP
#define CAF_OPENCL_ACTOR_FACADE_HPP

#include <ostream>
#include <iostream>
#include <algorithm>
#include <stdexcept>

#include "caf/all.hpp"

#include "caf/channel.hpp"
#include "caf/to_string.hpp"
#include "caf/intrusive_ptr.hpp"

#include "caf/detail/int_list.hpp"
#include "caf/detail/type_list.hpp"
#include "caf/detail/limited_vector.hpp"

#include "caf/opencl/global.hpp"
#include "caf/opencl/command.hpp"
#include "caf/opencl/program.hpp"
#include "caf/opencl/arguments.hpp"
#include "caf/opencl/smart_ptr.hpp"
#include "caf/opencl/opencl_err.hpp"
#include "caf/opencl/spawn_config.hpp"

namespace caf {
namespace opencl {

class opencl_metainfo;

//template <class Signature>
//class actor_facade;

template <class... Args>
class actor_facade : public abstract_actor {

public:
  using arg_types = detail::type_list<typename std::decay<Args>::type...>;

  using input_wrapped_types = detail::tl_filter<arg_types, is_input_arg>;
  using input_types = detail::tl_map<input_wrapped_types, extract_type>;
  using input_mapping = std::function<optional<message> (message&)>;

  using output_wrapped_types = detail::tl_filter<arg_types, is_output_arg>;
  using output_types = detail::tl_map<output_wrapped_types, extract_type>;
  using output_mapping = std::function<message(output_types...)>;

  using sized_types = detail::tl_filter<arg_types, requires_size_arg>;

  using evnt_vec = std::vector<cl_event>;
  using args_vec = std::vector<mem_ptr>;
  using size_vec = std::vector<size_t>;

  using command_type = command<actor_facade, output_types>;

  friend class command<actor_facade, output_types>;

  static intrusive_ptr<actor_facade>
  create(const program& prog, const char* kernel_name,
         const spawn_config& config,
         input_mapping map_args,
         output_mapping map_result,
         Args&&... arguments) {
    if (config.dimensions().empty()) {
      auto str = "OpenCL kernel needs at least 1 global dimension.";
      CAF_LOGF_ERROR(str);
      throw std::runtime_error(str);
    }
    auto check_vec = [&](const dim_vec& vec, const char* name) {
      if (! vec.empty() && vec.size() != config.dimensions().size()) {
        std::ostringstream oss;
        oss << name << " vector is not empty, but "
            << "its size differs from global dimensions vector's size";
        CAF_LOGF_ERROR(oss.str());
        throw std::runtime_error(oss.str());
      }
    };
    check_vec(config.offsets(), "offsets");
    check_vec(config.local_dimensions(), "local dimensions");
    kernel_ptr kernel;
    kernel.reset(v2get(CAF_CLF(clCreateKernel), prog.program_.get(),
                       kernel_name),
                 false);
    return new actor_facade(prog, kernel, config,
                            std::move(map_args), std::move(map_result),
                            std::forward_as_tuple(arguments...));
  }

  void enqueue(const actor_addr &sender, message_id mid, message content,
               execution_unit*) override {
    CAF_LOG_TRACE("");
    if (map_args_) {
      auto mapped = map_args_(content);
      if (! mapped) {
        return;
      }
      content = std::move(*mapped);
    }
    typename detail::il_indices<input_types>::type indices;
    if (! content.match_elements(input_types{})) {
      return;
    }
    response_promise handle{this->address(), sender, mid.response_id()};
    evnt_vec events;
    args_vec arguments;
    size_vec result_sizes;
    add_kernel_arguments(events, arguments, result_sizes, content, indices);
    auto cmd = make_counted<command_type>(handle, this,
                                          std::move(events),
                                          std::move(arguments),
                                          result_sizes,
                                          std::move(content));
    cmd->enqueue();
  }

private:
  actor_facade(const program& prog, kernel_ptr kernel,
               const spawn_config& config,
               input_mapping map_args, output_mapping map_result,
               std::tuple<Args...> args)
      : kernel_(kernel),
        program_(prog.program_),
        context_(prog.context_),
        queue_(prog.queue_),
        config_(config),
        map_args_(std::move(map_args)),
        map_result_(std::move(map_result)),
        argument_types_(args) {
    CAF_LOG_TRACE("id: " << this->id());
    default_output_size_ = std::accumulate(config_.dimensions().begin(),
                                           config_.dimensions().end(),
                                           size_t{1},
                                           std::multiplies<size_t>{});
  }

  void add_kernel_arguments(evnt_vec&, args_vec& arguments,
                            size_vec&, message&, unsigned) {
    for (cl_uint i = 0; i < arguments.size(); ++i) {
      v1callcl(CAF_CLF(clSetKernelArg), kernel_.get(), i,
               sizeof(cl_mem), static_cast<void*>(&arguments[i]));
    }
    clFlush(queue_.get());
  }

  template <long I, long... Is>
  void add_kernel_arguments(evnt_vec& events, args_vec& arguments,
                            size_vec& sizes, message& msg,
                            detail::int_list<I, Is...>) {
    using value_type = typename detail::tl_at<input_types, I>::type;
    auto value = msg.get_as<value_type>(I);
    create_buffer(std::get<I>(argument_types_), value, events, sizes,
                  arguments, msg);
    add_kernel_arguments(events, arguments, sizes, msg,
                         detail::int_list<Is...>{});
  }

  template <class T>
  void create_buffer(const in<T>&, T value, evnt_vec& events, size_vec&,
                     args_vec& arguments, message&) {
    size_t buffer_size = sizeof(T) * value.size();
    auto buffer = v2get(CAF_CLF(clCreateBuffer), context_.get(),
                        cl_mem_flags{CL_MEM_READ_ONLY}, buffer_size, nullptr);
    cl_event event = v1get<cl_event>(CAF_CLF(clEnqueueWriteBuffer),
                                     queue_.get(), buffer, cl_bool{CL_FALSE},
                                     cl_uint{0}, buffer_size, value.data());
    events.push_back(std::move(event));
    mem_ptr tmp;
    tmp.reset(buffer, false);
    arguments.push_back(tmp);
  }

  template <class T>
  void create_buffer(const in_out<T>&, T value, evnt_vec& events, size_vec&,
                     args_vec& arguments, message&) {
    size_t buffer_size = sizeof(T) * value.size();
    auto buffer = v2get(CAF_CLF(clCreateBuffer), context_.get(),
                        cl_mem_flags{CL_MEM_READ_WRITE}, buffer_size, nullptr);
    cl_event event = v1get<cl_event>(CAF_CLF(clEnqueueWriteBuffer),
                                     queue_.get(), buffer, cl_bool{CL_FALSE},
                                     cl_uint{0}, buffer_size, value.data());
    events.push_back(std::move(event));
    mem_ptr tmp;
    tmp.reset(buffer, false);
    arguments.push_back(tmp);
  }
  
  template <class T>
  void create_buffer(const out<T>& wrapper, T value, evnt_vec& events,
                     size_vec& sizes, args_vec& arguments, message& msg) {
    auto buffer_size = get_size_for_argument(wrapper.size_calculator_, msg,
                                             default_output_size_);
    auto buffer = v2get(CAF_CLF(clCreateBuffer), context_.get(),
                        cl_mem_flags{CL_MEM_READ_ONLY}, buffer_size, nullptr);
    cl_event event = v1get<cl_event>(CAF_CLF(clEnqueueWriteBuffer),
                                     queue_.get(), buffer, cl_bool{CL_FALSE},
                                     cl_uint{0}, buffer_size, value.data());
    events.push_back(std::move(event));
    mem_ptr tmp;
    tmp.reset(buffer, false);
    arguments.push_back(tmp);
    sizes.push_back(buffer_size);
  }

  size_t get_size_for_argument(dummy_size_calculator, const message&,
                               size_t default_size) {
    return default_size;
  }

  template <class Fun>
  size_t get_size_for_argument(Fun f, const message& m, size_t) {
    return m.apply(f);
  }

  kernel_ptr kernel_;
  program_ptr program_;
  context_ptr context_;
  command_queue_ptr queue_;
  spawn_config config_;
  input_mapping map_args_;
  output_mapping map_result_;
  std::tuple<Args&&...> argument_types_;
  size_t default_output_size_;
};

} // namespace opencl
} // namespace caf

#endif // CAF_OPENCL_ACTOR_FACADE_HPP
