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

template <class Signature>
class actor_facade;

template <typename... Args>
class actor_facade<Args...> : public abstract_actor {

public:
  using input_wrapped_types = detail::tl_filter<Args..., is_input_arg>;
  using input_types = detail::tl_map<input_wrapped_types, extract_type>; // was arg_types
  using input_mapping = std::function<optional<message> (message&)>; // was arg_mapping
  using output_wrapped_types = detail::tl_filter<Args..., is_output_arg>;
  using output_types = detail::tl_map<output_wrapped_types, extract_type>;
  using output_mapping = std::function<message(output_types...)>; // was result_mapping
  using sized_types = detail::tl_filter<Args..., requires_size_arg>;
  using evnt_vec = std::vector<cl_event>;
  using args_vec = std::vector<mem_ptr>;
  using size_vec = std::vector<size_t>;
  using command_type = command<actor_facade, output_types>;

  // does this work?
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
//    add_arguments_to_kernel<Ret>(events, arguments, result_sizes_,
//                                 content, indices);
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
        arguments_(args) {
    CAF_LOG_TRACE("id: " << this->id());
    default_output_size_ = std::accumulate(config_.dimensions().begin(),
                                           config_.dimensions().end(),
                                           size_t{1},
                                           std::multiplies<size_t>{});
  }

//  void add_arguments_to_kernel_rec(evnt_vec&, args_vec& arguments, message&,
//                                   detail::int_list<>) {
//    // rotate left (output buffer to the end)
//    std::rotate(arguments.begin(), arguments.begin() + 1, arguments.end());
//    for (cl_uint i = 0; i < arguments.size(); ++i) {
//      v1callcl(CAF_CLF(clSetKernelArg), kernel_.get(), i,
//               sizeof(cl_mem), static_cast<void*>(&arguments[i]));
//    }
//    clFlush(queue_.get());
//  }

//  template <long I, long... Is>
//  void add_arguments_to_kernel_rec(evnt_vec& events, args_vec& arguments,
//                                   message& msg, detail::int_list<I, Is...>) {
//    using value_type = typename detail::tl_at<input_types, I>::type;
//    auto& arg = msg.get_as<value_type>(I);
//    size_t buffer_size = sizeof(value_type) * arg.size();
//    auto buffer = v2get(CAF_CLF(clCreateBuffer), context_.get(),
//                        cl_mem_flags{CL_MEM_READ_ONLY}, buffer_size, nullptr);
//    cl_event event = v1get<cl_event>(CAF_CLF(clEnqueueWriteBuffer),
//                                     queue_.get(), buffer, cl_bool{CL_FALSE},
//                                     cl_uint{0}, buffer_size, arg.data());
//    events.push_back(std::move(event));
//    mem_ptr tmp;
//    tmp.reset(buffer, false);
//    arguments.push_back(tmp);
//    add_arguments_to_kernel_rec(events, arguments, msg,
//                                detail::int_list<Is...>{});
//  }

//  template <class R, class Token>
//  void add_arguments_to_kernel(evnt_vec& events, args_vec& arguments,
//                               size_t ret_size, message& msg, Token tk) {
//    arguments.clear();
//    auto buf = v2get(CAF_CLF(clCreateBuffer), context_.get(),
//                     cl_mem_flags{CL_MEM_WRITE_ONLY},
//                     sizeof(typename R::value_type) * ret_size, nullptr);
//    mem_ptr tmp;
//    tmp.reset(buf, false);
//    arguments.push_back(tmp);
//    add_arguments_to_kernel_rec(events, arguments, msg, tk);
//  }

  void create_mem_buffers(evnt_vec&, args_vec&, size_vec&,
                          message&, unsigned) {
    clFlush(queue_.get());
  }

  template <class T, class... Ts>
  void create_mem_buffers(evnt_vec& events, args_vec& args, size_vec& sizes,
                          message& msg, unsigned position) {
    using arg_type = typename T::type;
    auto arg = msg.get_as<arg_type>(position);
    create_buffer<arg_type>(T{}, arg, events, args, sizes, msg, position);
    create_mem_buffers<Ts...>(events, args, sizes, msg, ++position);
  }

  template <class T>
  void create_buffer(const in<T>&, T arg, evnt_vec& events, args_vec& args,
                     size_vec& sizes, message& msg, unsigned position) {

  }

  template <class T>
  void create_buffer(const in_out<T>&, T arg, evnt_vec& events, args_vec& args,
                     size_vec& sizes, message& msg, unsigned position) {

  }
  
  template <class T>
  void create_buffer(const out<T>& wrapper, T arg, evnt_vec& events, args_vec& args,
                     size_vec& sizes, message& msg, unsigned position) {
    auto size = get_size_for_argument(wrapper.size_calculator_, msg, default_output_size_);
    sizes.push_back(size);
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
  std::tuple<Args&&...> arguments_;
  size_t default_output_size_;
};

} // namespace opencl
} // namespace caf

#endif // CAF_OPENCL_ACTOR_FACADE_HPP
