/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file {{package_name}}/types.h
 * \brief Type-safe enumerations and constants.
 */

#ifndef {{package_name|upper}}_{{module_name|upper}}_TYPES_H_
#define {{package_name|upper}}_{{module_name|upper}}_TYPES_H_

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace {{package_name}} {
namespace {{module_name}} {

/*!
 * \brief Example enumeration.
 */
enum class ExampleEnum : int {
  /*! \brief Example value 0 */
  VALUE_0 = 0,
  /*! \brief Example value 1 */
  VALUE_1 = 1,
  /*! \brief Example value 2 */
  VALUE_2 = 2
};

/*!
 * \brief Memory types.
 */
enum class MemoryType : uint32_t {
  /*! \brief Host/DRAM memory */
  DRAM = 0,
  /*! \brief Device SRAM memory */
  SRAM = 1
};

/*!
 * \brief Buffer allocation alignment requirement in bytes.
 */
constexpr size_t kAllocAlignment = 64;

}  // namespace {{module_name}}
}  // namespace {{package_name}}

#endif  // {{package_name|upper}}_{{module_name|upper}}_TYPES_H_
