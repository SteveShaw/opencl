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

#ifndef CAF_SPAWN_CL_HPP
#define CAF_SPAWN_CL_HPP

#include <algorithm>
#include <functional>

#include "caf/optional.hpp"
#include "caf/actor_cast.hpp"

#include "caf/detail/limited_vector.hpp"

#include "caf/opencl/global.hpp"
#include "caf/opencl/arguments.hpp"
#include "caf/opencl/actor_facade.hpp"
#include "caf/opencl/spawn_config.hpp"
#include "caf/opencl/opencl_metainfo.hpp"

namespace caf {

namespace detail {

template <class... Ts>
struct cl_spawn_helper {
  using impl = opencl::actor_facade<Ts...>;
  using map_in_fun = std::function<optional<message> (message&)>;
  using map_out_fun = typename impl::output_mapping;

  actor operator()(const opencl::program& p, const char* fn,
                   const opencl::spawn_config& cfg, Ts&&... xs) const {
    return actor_cast<actor>(impl::create(p, fn, cfg,
                                          map_in_fun{}, map_out_fun{},
                                          std::forward<Ts>(xs)...));
  }

  actor operator()(const opencl::program& p, const char* fn,
                   const opencl::spawn_config& cfg,
                   map_in_fun map_input, map_out_fun map_output,
                   Ts&&... xs) const {
    return actor_cast<actor>(impl::create(p, fn, cfg, std::move(map_input),
                                          std::move(map_output),
                                          std::forward<Ts>(xs)...));
  }
};

} // namespace detail

/// Creates a new actor facade for an OpenCL kernel that invokes
/// the function named `fname` from `prog`.
/// @throws std::runtime_error if more than three dimensions are set,
///                            `dims.empty()`, or `clCreateKernel` failed.
template <class T, class... Ts>
typename std::enable_if<
  opencl::is_output_arg<T>::value || opencl::is_input_arg<T>::value,
  actor
>::type
spawn_cl(const opencl::program& prog,
               const char* fname,
               const opencl::spawn_config& config,
               T x,
               Ts... xs) {
  detail::cl_spawn_helper<T, Ts...> f;
  return f(prog, fname, config, std::move(x), std::move(xs)...);
}

/// Compiles `source` and creates a new actor facade for an OpenCL kernel
/// that invokes the function named `fname`.
/// @throws std::runtime_error if more than three dimensions are set,
///                            <tt>dims.empty()</tt>, a compilation error
///                            occured, or @p clCreateKernel failed.
template <class T, class... Ts>
typename std::enable_if<
  opencl::is_output_arg<T>::value || opencl::is_input_arg<T>::value,
  actor
>::type
spawn_cl(const char* source,
               const char* fname,
               const opencl::spawn_config& config,
               T x,
               Ts... xs) {
  detail::cl_spawn_helper<T, Ts...> f;
  return f(opencl::program::create(source), fname, config,
           std::move(x), std::move(xs)...);
}

/// Creates a new actor facade for an OpenCL kernel that invokes
/// the function named `fname` from `prog`.
/// @throws std::runtime_error if more than three dimensions are set,
///                            `dims.empty()`, or `clCreateKernel` failed.
template <class Fun, class... Ts>
actor spawn_cl(const opencl::program& prog,
               const char* fname,
               const opencl::spawn_config& config,
               std::function<optional<message> (message&)> map_args,
               Fun map_result,
               Ts... xs) {
  detail::cl_spawn_helper<Ts...> f;
  return f(prog, fname, config, std::move(map_args), std::move(map_result),
           std::forward<Ts>(xs)...);
}

/// Compiles `source` and creates a new actor facade for an OpenCL kernel
/// that invokes the function named `fname`.
/// @throws std::runtime_error if more than three dimensions are set,
///                            <tt>dims.empty()</tt>, a compilation error
///                            occured, or @p clCreateKernel failed.
template <class Fun, class... Ts>
actor spawn_cl(const char* source,
               const char* fname,
               const opencl::spawn_config& config,
               std::function<optional<message> (message&)> map_args,
               Fun map_result,
               Ts... xs) {
  detail::cl_spawn_helper<Ts...> f;
  return f(opencl::program::create(source), fname, config,
           std::move(map_args), std::move(map_result),
           std::forward<Ts>(xs)...);
}

} // namespace caf

#endif // CAF_SPAWN_CL_HPP
