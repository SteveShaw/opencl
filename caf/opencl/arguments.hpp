/******************************************************************************
 *                       ____    _    _____                                   *
 *                      / ___|  / \  |  ___|    C++                           *
 *                     | |     / _ \ | |_       Actor                         *
 *                     | |___ / ___ \|  _|      Framework                     *
 *                      \____/_/   \_|_|                                      *
 *                                                                            *
 * Copyright (C) 2011 - 2015                                                  *
 * Dominik Charousset <dominik.charousset (at) haw-hamburg.de>                *
 * Raphael Hiesgen <raphael.hiesgen (at) haw-hamburg.de>                      *
 *                                                                            *
 * Distributed under the terms and conditions of the BSD 3-Clause License or  *
 * (at your option) under the terms and conditions of the Boost Software      *
 * License 1.0. See accompanying files LICENSE and LICENSE_ALTERNATIVE.       *
 *                                                                            *
 * If you did not receive a copy of the license files, see                    *
 * http://opensource.org/licenses/BSD-3-Clause and                            *
 * http://www.boost.org/LICENSE_1_0.txt.                                      *
 ******************************************************************************/

#ifndef CAF_OPENCL_ARGUMENTS
#define CAF_OPENCL_ARGUMENTS

namespace caf {
namespace opencl {

/// Mark an a spawn_cl template argument as input only
template <typename Arg>
struct in {
  using arg_type = typename std::decay<Arg>::type;
};

/// Mark an a spawn_cl template argument as input and output
template <typename Arg>
struct in_out {
  using arg_type = typename std::decay<Arg>::type;
};

/// Mark an a spawn_cl template argument as output with optional size,
/// set to 0 the size will be set the number of work items
template <typename Arg, size_t Size = 0>
struct out {
  using arg_type = typename std::decay<Arg>::type;
  size_t size = Size;
};

/// filter type lists for input arguments, in and in_out
template <class T>
struct is_input_arg : std::false_type {};

template <class T>
struct is_input_arg<in<T>> : std::true_type {};

template <class T>
struct is_input_arg<in_out<T>> : std::true_type {};

/// filter type lists for output arguments, in_out and out
template <class T>
struct is_output_arg : std::false_type {};

template <class T>
struct is_output_arg<out<T>> : std::true_type {};

template <class T>
struct is_output_arg<in_out<T>> : std::true_type {};

// filter for arguments that require size, in this case only out
template <class T>
struct requires_size_arg : std::false_type {};

template <class T>
struct requires_size_arg<out<T>> : std::true_type {};

// extract types
template <class T>
struct extract_type { };


template <class T>
struct extract_type<in<T>> { using type = T; };


template <class T>
struct extract_type<in_out<T>> { using type = T; };


template <class T>
struct extract_type<out<T>> { using type = T; };


// extract the sizes of output arguments into a vector
template<class T, class... Ts>
void create_vector_with_sizes(std::vector<size_t>& sizes, size_t default_size) {
  auto size = T::size;
  sizes.push_back(size == 0 ? default_size : size);
  create_vector_with_sizes<Ts...>(sizes);
}

void create_vector_with_sizes(std::vector<size_t>&, size_t) {
  return;
}

} // namespace opencl
} // namespace caf

#endif // CAF_OPENCL_ARGUMENTS