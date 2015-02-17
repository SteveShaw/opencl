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

#ifndef CAF_OPENCL_PROGRAM_HPP
#define CAF_OPENCL_PROGRAM_HPP

#include <memory>

#include "caf/opencl/global.hpp"
#include "caf/opencl/smart_ptr.hpp"

namespace caf {
namespace opencl {

template <typename Signature>
class actor_facade;

/**
 * @brief A wrapper for OpenCL's cl_program.
 */
class program {

  template <typename Signature>
  friend class actor_facade;

 public:
  /**
   * @brief Factory method, that creates a caf::opencl::program
   *        from a given @p kernel_source.
   * @returns A program object.
   */
  static program create(const char* kernel_source,
                        const char* options = nullptr, uint32_t device_id = 0);

 private:
  program(context_ptr context, command_queue_ptr queue, program_ptr program);

  context_ptr m_context;
  program_ptr m_program;
  command_queue_ptr m_queue;
};

} // namespace opencl
} // namespace caf

#endif // CAF_OPENCL_PROGRAM_HPP
