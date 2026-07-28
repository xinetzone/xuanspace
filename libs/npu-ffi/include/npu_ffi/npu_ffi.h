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

#pragma once

/*!
 * \file npu_ffi/npu_ffi.h
 * \brief Main header file for NPU FFI - include this to get all VTA APIs.
 *
 * This header pulls in the common platform utilities, logging system,
 * and the VTA runtime API. For reusable components (logging, platform
 * detection, DLL export macros), include npu_ffi/common.h directly.
 */

#include "npu_ffi/common.h"
#include "npu_ffi/logging.h"
#include "npu_ffi/vta/runtime.h"

/*!
 * \brief NPU FFI version major number.
 */
#define NPU_FFI_VERSION_MAJOR 0

/*!
 * \brief NPU FFI version minor number.
 */
#define NPU_FFI_VERSION_MINOR 1

/*!
 * \brief NPU FFI version patch number.
 */
#define NPU_FFI_VERSION_PATCH 0

/*!
 * \brief Helper macro for stringifying version numbers.
 */
#define NPU_FFI_STR_HELPER(x) #x

/*!
 * \brief Stringify macro.
 */
#define NPU_FFI_STR(x) NPU_FFI_STR_HELPER(x)

/*!
 * \brief NPU FFI version string in "major.minor.patch" format.
 */
#define NPU_FFI_VERSION_STRING \
  NPU_FFI_STR(NPU_FFI_VERSION_MAJOR) "." \
  NPU_FFI_STR(NPU_FFI_VERSION_MINOR) "." \
  NPU_FFI_STR(NPU_FFI_VERSION_PATCH)
