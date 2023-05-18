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
 * \file sdaa_module.cc
 */
// #include "sdaa_module.h"

// #include <sdaa.h>
#include <sdaa_runtime.h>
#include <tvm/runtime/registry.h>

#include <array>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "../file_utils.h"
#include "../meta_data.h"
#include "../pack_args.h"
#include "../thread_storage_scope.h"
#include "sdaa_common.h"
#include "sdaa_module.h"

#define SDAA_SUCCESS 0
// #define SDAA_ERROR_DEINITIALIZED 4
namespace tvm {
namespace runtime {

// Module to support thread-safe multi-SWAI execution.
// SDmodule is a per-SWAI module
// The runtime will contain a per-device module table
// The modules will be lazily loaded ???
class SDAAModuleNode : public runtime::ModuleNode {
 public:
  explicit SDAAModuleNode(std::string data, std::string fmt,
                          std::unordered_map<std::string, FunctionInfo> fmap,
                          std::string sdaa_source)
      : data_(data), fmt_(fmt), fmap_(fmap), sdaa_source_(sdaa_source) {
        int device_id;
        SDAA_CALL(sdaaGetDevice(&device_id));
        VLOG(2) << "SDAAModuleNode's input data: " << data.c_str();
        SDAA_CALL(sdaaModuleLoad(&(module_[device_id]), data_.c_str()));
    // std::fill(module_.begin(), module_.end(), nullptr);

  }
  // destructor
  ~SDAAModuleNode() {
    for (size_t i = 0; i < module_.size(); ++i) {
      if (module_[i] != nullptr) {
        SDAA_CALL(sdaaSetDevice(static_cast<int>(i)));
        SDAA_CALL(sdaaModuleUnload(module_[i]));
      }
    }
  }

  const char* type_key() const final { return "sdaa"; }

  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) final;

  void SaveToFile(const std::string& file_name, const std::string& format) final {
    std::string fmt = GetFileFormat(file_name, format);
    std::string meta_file = GetMetaFilePath(file_name);
    if (fmt == "swcu") {
      ICHECK_NE(sdaa_source_.length(), 0);
      SaveMetaDataToFile(meta_file, fmap_);
      SaveBinaryToFile(file_name, sdaa_source_);
    } else {
      ICHECK_EQ(fmt, fmt_) << "Can only save to format=" << fmt_;
      SaveMetaDataToFile(meta_file, fmap_);
      SaveBinaryToFile(file_name, data_);
    }
  }

  void SaveToBinary(dmlc::Stream* stream) final {
    stream->Write(fmt_);
    stream->Write(fmap_);
    stream->Write(data_);
  }

  std::string GetSource(const std::string& format) final {
    if (format == fmt_) return data_;
    if (sdaa_source_.length() != 0) {
      return sdaa_source_;
    } else {
      if (fmt_ == "so") return data_;
      return "";
    }
  }

  // zly: sdaa doesn't have something like primary context.
  // get a sdaaFunction_t from primary context in device_id
  sdaaFunction_t GetFunc(int device_id, const std::string& func_name) {
    std::lock_guard<std::mutex> lock(mutex_);
    // must recheck under the lock scope
    // zly: this may be the way of lazy loading, i.e., only when the function is called, 
    // the module is loaded.
    if (module_[device_id] == nullptr) {
      SDAA_CALL(sdaaModuleLoad(&(module_[device_id]), data_.c_str()));
    }
    sdaaFunction_t func;
    sdaaError_t result = sdaaModuleGetFunction(&func, module_[device_id], func_name.c_str());
    if (result != SDAA_SUCCESS) {
      const char* msg = sdaaGetErrorName(result);
      LOG(FATAL) << "SDAAError: sdaaModuleGetFunction " << func_name << " failed with error: " << msg;
    }
    return func;
  }
  
  // get a global var from primary context in device_id
  // sdaaPointerAttributes GetGlobal(int device_id, const std::string& global_name, size_t expect_nbytes) {
  //   std::lock_guard<std::mutex> lock(mutex_);
  //   // must recheck under the lock scope
  //   if (module_[device_id] == nullptr) {
  //     SDAA_CALL(sdaaModuleLoad(&(module_[device_id]), data_.c_str()));
  //   }
  //   sdaaPointerAttributes global;
  //   size_t nbytes;

  //   // zly: only support to check the information of a pointer instead of a global name.
  //   sdaaModule_t result = sdaaPointerGetAttributes(&global, &nbytes, module_[device_id], global_name.c_str());
  //   ICHECK_EQ(nbytes, expect_nbytes);
  //   if (result != SDAA_SUCCESS) {
  //     const char* msg;
  //     cuGetErrorName(result, &msg);
  //     LOG(FATAL) << "SDAAError: SDmoduleGetGlobal " << global_name << " failed with error: " << msg;
  //   }
  //   return global;
  // }

 private:
  // the binary data
  std::string data_;
  // The format
  std::string fmt_;
  // function information table.
  std::unordered_map<std::string, FunctionInfo> fmap_;
  // The sdaa source.
  std::string sdaa_source_;
  // the internal modules per SWAI, to be lazily initialized.
  std::array<sdaaModule_t, kMaxNumSWAIs> module_;
  // internal mutex when updating the module
  std::mutex mutex_;
};

// a wrapped function class to get packed func.
class SDAAWrappedFunc {
 public:
  // initialize the SDAA function.
  void Init(SDAAModuleNode* m, ObjectPtr<Object> sptr, const std::string& func_name,
            size_t num_void_args, const std::vector<std::string>& launch_param_tags) {
    m_ = m;
    sptr_ = sptr;
    func_name_ = func_name;
    std::fill(fcache_.begin(), fcache_.end(), nullptr);
    launch_param_config_.Init(num_void_args, launch_param_tags);
  }
  // invoke the function with void arguments
  void operator()(TVMArgs args, TVMRetValue* rv, void** void_args) const {
    int device_id;
    SDAA_CALL(sdaaGetDevice(&device_id));
    ThreadWorkLoad wl = launch_param_config_.Extract(args);

    if (fcache_[device_id] == nullptr) {
      fcache_[device_id] = m_->GetFunc(device_id, func_name_);
      // if (wl.dyn_shmem_size >= (48 << 10)) {
      //   // Assumption: dyn_shmem_size doesn't change across different invocations of
      //   // fcache_[device_id]
      //   sdaaModule_t result = cuFuncSetAttribute(
      //       fcache_[device_id], CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, wl.dyn_shmem_size);
      //   if (result != SDAA_SUCCESS) {
      //     LOG(FATAL) << "Failed to set the allowed dynamic shared memory size to "
      //                << wl.dyn_shmem_size;
      //   }
      // }
    }
    sdaaStream_t strm = static_cast<sdaaStream_t>(SDAAThreadEntry::ThreadLocal()->stream);
    //第二个参数arrayNum [in]：所需核组个数，取值目前只支持1核组
    sdaaError_t result = sdaaModuleLaunchKernel(fcache_[device_id], 1, strm, void_args);
    if (result != SDAA_SUCCESS) {
      const char* msg = sdaaGetErrorName(result);
      std::ostringstream os;
      os << "SDAALaunch Error: " << msg << "\n";
      std::string sdaa = m_->GetSource("");
      if (sdaa.length() != 0) {
        os << "// func_name=" << func_name_ << "\n"
           << "// SDAA Source\n"
           << "// -----------\n"
           << sdaa;
      }
      LOG(FATAL) << os.str();
    }
  }

 private:
  // internal module
  SDAAModuleNode* m_;
  // the resource holder
  ObjectPtr<Object> sptr_;
  // The name of the function.
  std::string func_name_;
  // Device function cache per device.
  // mark as mutable, to enable lazy initialization
  mutable std::array<sdaaFunction_t, kMaxNumSWAIs> fcache_;
  // launch parameters configuration
  LaunchParamConfig launch_param_config_;
};

// class SDAAPrepGlobalBarrier {
//  public:
//   SDAAPrepGlobalBarrier(SDAAModuleNode* m, ObjectPtr<Object> sptr) : m_(m), sptr_(sptr) {
//     std::fill(pcache_.begin(), pcache_.end(), 0);
//   }

//   void operator()(const TVMArgs& args, TVMRetValue* rv) const {
//     int device_id;
//     SDAA_CALL(sdaaGetDevice(&device_id));
//     if (pcache_[device_id] == 0) {
//       pcache_[device_id] =
//           m_->GetGlobal(device_id, runtime::symbol::tvm_global_barrier_state, sizeof(unsigned));
//     }
//     SDAA_DRIVER_CALL(sdaaMemset(pcache_[device_id], 0, 1));
//   }

//  private:
//   // internal module
//   SDAAModuleNode* m_;
//   // the resource holder
//   ObjectPtr<Object> sptr_;
//   // mark as mutable, to enable lazy initialization
//   // mutable std::array<SDdeviceptr, kMaxNumGPUs> pcache_;
// };

PackedFunc SDAAModuleNode::GetFunction(const std::string& name,
                                       const ObjectPtr<Object>& sptr_to_self) {
  ICHECK_EQ(sptr_to_self.get(), this);
  ICHECK_NE(name, symbol::tvm_module_main) << "Device function do not have main";
  ICHECK_NE(name, symbol::tvm_prepare_global_barrier) << "Device do not support global barrier";
  // if (name == symbol::tvm_prepare_global_barrier) {
  //   return PackedFunc(SDAAPrepGlobalBarrier(this, sptr_to_self));
  // }
  auto it = fmap_.find(name);
  if (it == fmap_.end()) return PackedFunc();
  const FunctionInfo& info = it->second;
  SDAAWrappedFunc f;
  f.Init(this, sptr_to_self, name, info.arg_types.size(), info.launch_param_tags);
  return PackFuncVoidAddr(f, info.arg_types);
}

Module SDAAModuleCreate(std::string data, std::string fmt,
                        std::unordered_map<std::string, FunctionInfo> fmap,
                        std::string sdaa_source) {
  auto n = make_object<SDAAModuleNode>(data, fmt, fmap, sdaa_source);
  return Module(n);
}

// Load module from module.
Module SDAAModuleLoadFile(const std::string& file_name, const std::string& format) {
  std::string data;
  std::unordered_map<std::string, FunctionInfo> fmap;
  std::string fmt = GetFileFormat(file_name, format);
  std::string meta_file = GetMetaFilePath(file_name);
  LoadBinaryFromFile(file_name, &data);
  LoadMetaDataFromFile(meta_file, &fmap);
  return SDAAModuleCreate(data, fmt, fmap, std::string());
}

Module SDAAModuleLoadBinary(void* strm) {
  dmlc::Stream* stream = static_cast<dmlc::Stream*>(strm);
  std::string data;
  std::unordered_map<std::string, FunctionInfo> fmap;
  std::string fmt;
  stream->Read(&fmt);
  stream->Read(&fmap);
  stream->Read(&data);
  return SDAAModuleCreate(data, fmt, fmap, std::string());
}

// TVM_REGISTER_GLOBAL("runtime.module.loadfile_cubin").set_body_typed(SDAAModuleLoadFile);

TVM_REGISTER_GLOBAL("runtime.module.loadfile_bin").set_body_typed(SDAAModuleLoadFile);

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_sdaa").set_body_typed(SDAAModuleLoadBinary);
}  // namespace runtime
}  // namespace tvm
