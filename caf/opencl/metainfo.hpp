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

#ifndef CAF_METAINFO_HPP
#define CAF_METAINFO_HPP

#include <atomic>
#include <vector>
#include <algorithm>
#include <functional>

#include "caf/all.hpp"
#include "caf/maybe.hpp"
#include "caf/config.hpp"
#include "caf/actor_system.hpp"

#include "caf/opencl/device.hpp"
#include "caf/opencl/global.hpp"
#include "caf/opencl/program.hpp"
#include "caf/opencl/platform.hpp"
#include "caf/opencl/smart_ptr.hpp"
#include "caf/opencl/actor_facade.hpp"

namespace caf {

namespace detail {

struct tuple_construct { };

template <class... Ts>
struct cl_spawn_helper {
  using impl = opencl::actor_facade<Ts...>;
  using map_in_fun = std::function<maybe<message> (message&)>;
  using map_out_fun = typename impl::output_mapping;

  actor operator()(const opencl::program& p, const char* fn,
                   const opencl::spawn_config& cfg, Ts&&... xs) const {
    return actor_cast<actor>(impl::create(p, fn, cfg,
                                          map_in_fun{}, map_out_fun{},
                                          std::move(xs)...));
  }
  actor operator()(const opencl::program& p, const char* fn,
                   const opencl::spawn_config& cfg,
                   map_in_fun map_input, map_out_fun map_output,
                   Ts&&... xs) const {
    return actor_cast<actor>(impl::create(p, fn, cfg, std::move(map_input),
                                          std::move(map_output),
                                          std::move(xs)...));
  }
  // used by the deprecated spawn_helper
  template <class Tuple, long... Is>
  actor operator()(tuple_construct,
                   const opencl::program& p, const char* fn,
                   const opencl::spawn_config& cfg,
                   Tuple&& xs,
                   detail::int_list<Is...>) const {
    return actor_cast<actor>(impl::create(p, fn, cfg,
                                          map_in_fun{}, map_out_fun{},
                                          std::move(std::get<Is>(xs))...));
  }
  template <class Tuple, long... Is>
  actor operator()(tuple_construct,
                   const opencl::program& p, const char* fn,
                   const opencl::spawn_config& cfg,
                   map_in_fun map_input, map_out_fun map_output,
                   Tuple&& xs,
                   detail::int_list<Is...>) const {
    return actor_cast<actor>(impl::create(p, fn, cfg,std::move(map_input),
                                          std::move(map_output),
                                          std::move(std::get<Is>(xs))...));
  }
};

} // namespace detail

namespace opencl {

class metainfo : public actor_system::module {

  friend class program;
  friend class actor_system;
  friend command_queue_ptr get_command_queue(uint32_t id);

public:
  /// Get a list of all available devices. This is depricated, use the more specific
  /// get_deivce and get_deivce_if functions.
  /// (Returns only devices of the first discovered platform).
  const std::vector<device>& get_devices() const CAF_DEPRECATED;
  /// Get the device with id. These ids are assigned sequientally to all available devices.
  const maybe<const device&> get_device(size_t id = 0) const;
  /// Get the first device that satisfies the predicate.
  /// The predicate should accept a `const device&` and return a bool;
  template <class UnaryPredicate>
  const maybe<const device&> get_device_if(UnaryPredicate p) const {
    for (auto& pl : platforms_) {
      for (auto& dev : pl.get_devices()) {
        if (p(dev))
          return dev;
      }
    }
    return none;
  }

  void start() override;
  void stop() override;
  void init(actor_system_config&) override;

  id_t id() const override;

  void* subtype_ptr() override;

  static actor_system::module* make(actor_system&, detail::type_list<>);

  // OpenCL functionality

  /// @brief Factory method, that creates a caf::opencl::program
  ///        from a given @p kernel_source.
  /// @returns A program object.
  program create_program(const char* kernel_source,
                         const char* options = nullptr, uint32_t device_id = 0);

  /// @brief Factory method, that creates a caf::opencl::program
  ///        from a given @p kernel_source.
  /// @returns A program object.
  program create_program(const char* kernel_source,
                         const char* options, const device& dev);

  /// Creates a new actor facade for an OpenCL kernel that invokes
  /// the function named `fname` from `prog`.
  /// @throws std::runtime_error if more than three dimensions are set,
  ///                            `dims.empty()`, or `clCreateKernel` failed.
  template <class T, class... Ts>
  typename std::enable_if<
    opencl::is_opencl_arg<T>::value,
    actor
  >::type
  spawn(const opencl::program& prog,
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
    opencl::is_opencl_arg<T>::value,
    actor
  >::type
  spawn(const char* source,
        const char* fname,
        const opencl::spawn_config& config,
        T x,
        Ts... xs) {
    detail::cl_spawn_helper<T, Ts...> f;
    return f(create_program(source), fname, config,
             std::move(x), std::move(xs)...);
  }

  /// Creates a new actor facade for an OpenCL kernel that invokes
  /// the function named `fname` from `prog`.
  /// @throws std::runtime_error if more than three dimensions are set,
  ///                            `dims.empty()`, or `clCreateKernel` failed.
  template <class Fun, class... Ts>
  actor spawn(const opencl::program& prog,
              const char* fname,
              const opencl::spawn_config& config,
              std::function<maybe<message> (message&)> map_args,
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
  actor spawn(const char* source,
              const char* fname,
              const opencl::spawn_config& config,
              std::function<maybe<message> (message&)> map_args,
              Fun map_result,
              Ts... xs) {
    detail::cl_spawn_helper<Ts...> f;
    return f(create_program(source), fname, config,
             std::move(map_args), std::move(map_result),
             std::forward<Ts>(xs)...);
  }

protected:
  metainfo(actor_system& sys);

private:
  actor_system& system_;
  std::vector<platform> platforms_;
};

} // namespace opencl
} // namespace caf

#endif // CAF_METAINFO_HPP
