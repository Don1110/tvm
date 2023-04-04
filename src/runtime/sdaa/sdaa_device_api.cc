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
 * \file cuda_device_api.cc
 * \brief GPU specific API
 */
// #include <sdaa.h>
#include <sdaa_runtime.h>
#include <dmlc/thread_local.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/profiling.h>
#include <tvm/runtime/registry.h>

#include <cstring>

#include "sdaa_common.h"

namespace tvm {
namespace runtime {

class CUDADeviceAPI final : public DeviceAPI {    
 public:
  void SetDevice(Device dev) final { SDAA_CALL(sdaaSetDevice(dev.device_id)); }
  void GetAttr(Device dev, DeviceAttrKind kind, TVMRetValue* rv) final {
    int value = 0;
    switch (kind) {
      case kExist:
        // value = (sdaaDeviceGetAttribute(&value, cudaDevAttrMaxThreadsPerBlock, dev.device_id) ==
        //          sdaaSuccess);
        break;
      case kMaxThreadsPerBlock: {
        // SDAA_CALL(sdaaDeviceGetAttribute(&value, cudaDevAttrMaxThreadsPerBlock, dev.device_id));
        // break;
      }
      case kWarpSize: {
        // SDAA_CALL(sdaaDeviceGetAttribute(&value, cudaDevAttrWarpSize, dev.device_id));
        // break;
      }
      case kMaxSharedMemoryPerBlock: {
        // SDAA_CALL(
        //     sdaaDeviceGetAttribute(&value, cudaDevAttrMaxSharedMemoryPerBlock, dev.device_id));
        break;
      }
      case kComputeVersion: {
        std::ostringstream os;
        // SDAA_CALL(sdaaDeviceGetAttribute(&value, cudaDevAttrComputeCapabilityMajor, dev.device_id));
        os << value << ".";
        // SDAA_CALL(sdaaDeviceGetAttribute(&value, cudaDevAttrComputeCapabilityMinor, dev.device_id));
        os << value;
        *rv = os.str();
        return;
      }
      case kDeviceName: {
        std::string name(256, 0);
        // SDAA_DRIVER_CALL(sdDeviceGetName(&name[0], name.size(), dev.device_id));
        name.resize(strlen(name.c_str()));
        *rv = std::move(name);
        return;
      }
      case kMaxClockRate: {
        // SDAA_CALL(sdaaDeviceGetAttribute(&value, cudaDevAttrClockRate, dev.device_id));
        break;
      }
      case kMultiProcessorCount: {
        // SDAA_CALL(sdaaDeviceGetAttribute(&value, cudaDevAttrMultiProcessorCount, dev.device_id));
        break;
      }
      case kMaxThreadDimensions: {
        int dims[3];
        // SDAA_CALL(sdaaDeviceGetAttribute(&dims[0], cudaDevAttrMaxBlockDimX, dev.device_id));
        // SDAA_CALL(sdaaDeviceGetAttribute(&dims[1], cudaDevAttrMaxBlockDimY, dev.device_id));
        // SDAA_CALL(sdaaDeviceGetAttribute(&dims[2], cudaDevAttrMaxBlockDimZ, dev.device_id));

        std::stringstream ss;  // use json string to return multiple int values;
        ss << "[" << dims[0] << ", " << dims[1] << ", " << dims[2] << "]";
        *rv = ss.str();
        return;
      }
      case kMaxRegistersPerBlock: {
        // SDAA_CALL(sdaaDeviceGetAttribute(&value, cudaDevAttrMaxRegistersPerBlock, dev.device_id));
        break;
      }
      case kGcnArch:
        return;
      case kApiVersion: {
        // *rv = SDAA_VERSION;
        return;
      }
      case kDriverVersion:
        return;
    }
    *rv = value;
  }
  void* AllocDataSpace(Device dev, size_t nbytes, size_t alignment, DLDataType type_hint) final {
    ICHECK_EQ(256 % alignment, 0U) << "CUDA space is aligned at 256 bytes";
    void* ret;
    if (dev.device_type == kDLSDAAHost) {
      VLOG(1) << "allocating " << nbytes << "bytes on host";
      SDAA_CALL(sdaaMallocHost(&ret, nbytes));
    } else {
      SDAA_CALL(sdaaSetDevice(dev.device_id));
      size_t free_mem, total_mem;
      SDAA_CALL(sdaaMemGetInfo(&free_mem, &total_mem));
      VLOG(1) << "allocating " << nbytes << " bytes on device, with " << free_mem
              << " bytes currently free out of " << total_mem << " bytes available";
      SDAA_CALL(sdaaMalloc(&ret, nbytes));
    }
    return ret;
  }

  void FreeDataSpace(Device dev, void* ptr) final {
    if (dev.device_type == kDLSDAAHost) {
      VLOG(1) << "freeing host memory";
      SDAA_CALL(sdaaFreeHost(ptr));
    } else {
      SDAA_CALL(sdaaSetDevice(dev.device_id));
      VLOG(1) << "freeing device memory";
      SDAA_CALL(sdaaFree(ptr));
    }
  }

 protected:
  void CopyDataFromTo(const void* from, size_t from_offset, void* to, size_t to_offset, size_t size,
                      Device dev_from, Device dev_to, DLDataType type_hint,
                      TVMStreamHandle stream) final {
    sdaaStream_t cu_stream = static_cast<sdaaStream_t>(stream);
    from = static_cast<const char*>(from) + from_offset;
    to = static_cast<char*>(to) + to_offset;

    if (dev_from.device_type == kDLSDAAHost) {
      dev_from.device_type = kDLCPU;
    }

    if (dev_to.device_type == kDLSDAAHost) {
      dev_to.device_type = kDLCPU;
    }

    // In case there is a copy from host mem to host mem */
    if (dev_to.device_type == kDLCPU && dev_from.device_type == kDLCPU) {
      memcpy(to, from, size);
      return;
    }

    if (dev_from.device_type == kDLSDAA && dev_to.device_type == kDLSDAA) {
      SDAA_CALL(sdaaSetDevice(dev_from.device_id));
      if (dev_from.device_id == dev_to.device_id) {
        GPUCopy(from, to, size, sdaaMemcpyDeviceToDevice, cu_stream);
      } else {
        // cudaMemcpyPeerAsync(to, dev_to.device_id, from, dev_from.device_id, size, cu_stream);
      }
    } else if (dev_from.device_type == kDLSDAA && dev_to.device_type == kDLCPU) {
      SDAA_CALL(sdaaSetDevice(dev_from.device_id));
      GPUCopy(from, to, size, sdaaMemcpyDeviceToHost, cu_stream);
    } else if (dev_from.device_type == kDLCPU && dev_to.device_type == kDLSDAA) {
      SDAA_CALL(sdaaSetDevice(dev_to.device_id));
      GPUCopy(from, to, size, sdaaMemcpyHostToDevice, cu_stream);
    } else {
      LOG(FATAL) << "expect copy from/to GPU or between GPU";
    }
  }

 public:
  TVMStreamHandle CreateStream(Device dev) {
    SDAA_CALL(sdaaSetDevice(dev.device_id));
    sdaaStream_t retval;
    SDAA_CALL(sdaaStreamCreate(&retval));
    return static_cast<TVMStreamHandle>(retval);
  }

  void FreeStream(Device dev, TVMStreamHandle stream) {
    SDAA_CALL(sdaaSetDevice(dev.device_id));
    sdaaStream_t cu_stream = static_cast<sdaaStream_t>(stream);
    SDAA_CALL(sdaaStreamDestroy(cu_stream));
  }

  void SyncStreamFromTo(Device dev, TVMStreamHandle event_src, TVMStreamHandle event_dst) {
    SDAA_CALL(sdaaSetDevice(dev.device_id));
    sdaaStream_t src_stream = static_cast<sdaaStream_t>(event_src);
    sdaaStream_t dst_stream = static_cast<sdaaStream_t>(event_dst);
    sdaaEvent_t evt;
    SDAA_CALL(sdaaEventCreate(&evt));
    SDAA_CALL(sdaaEventRecord(evt, src_stream));
    SDAA_CALL(sdaaStreamWaitEvent(dst_stream, evt, 0));
    SDAA_CALL(sdaaEventDestroy(evt));
  }

  void StreamSync(Device dev, TVMStreamHandle stream) final {
    SDAA_CALL(sdaaSetDevice(dev.device_id));
    SDAA_CALL(sdaaStreamSynchronize(static_cast<sdaaStream_t>(stream)));
  }

  void SetStream(Device dev, TVMStreamHandle stream) final {
    SDAAThreadEntry::ThreadLocal()->stream = static_cast<sdaaStream_t>(stream);
  }

  void* AllocWorkspace(Device dev, size_t size, DLDataType type_hint) final {
    return SDAAThreadEntry::ThreadLocal()->pool.AllocWorkspace(dev, size);
  }

  void FreeWorkspace(Device dev, void* data) final {
    SDAAThreadEntry::ThreadLocal()->pool.FreeWorkspace(dev, data);
  }

  static CUDADeviceAPI* Global() {
    // NOTE: explicitly use new to avoid exit-time destruction of global state
    // Global state will be recycled by OS as the process exits.
    static auto* inst = new CUDADeviceAPI();
    return inst;
  }

 private:
  static void GPUCopy(const void* from, void* to, size_t size, sdaaMemcpyKind kind,
                      sdaaStream_t stream) {
    if (stream != nullptr) {
      SDAA_CALL(sdaaMemcpyAsync(to, from, size, kind, stream));
    } else {
      SDAA_CALL(sdaaMemcpy(to, from, size, kind));
    }
  }
};

typedef dmlc::ThreadLocalStore<SDAAThreadEntry> CUDAThreadStore;

SDAAThreadEntry::SDAAThreadEntry() : pool(kDLSDAA, CUDADeviceAPI::Global()) {}

SDAAThreadEntry* SDAAThreadEntry::ThreadLocal() { return CUDAThreadStore::Get(); }

TVM_REGISTER_GLOBAL("device_api.cuda").set_body([](TVMArgs args, TVMRetValue* rv) {
  DeviceAPI* ptr = CUDADeviceAPI::Global();
  *rv = static_cast<void*>(ptr);
});

TVM_REGISTER_GLOBAL("device_api.cuda_host").set_body([](TVMArgs args, TVMRetValue* rv) {
  DeviceAPI* ptr = CUDADeviceAPI::Global();
  *rv = static_cast<void*>(ptr);
});

class CUDATimerNode : public TimerNode {
 public:
  virtual void Start() {
    // This initial cudaEventRecord is sometimes pretty slow (~100us). Does
    // cudaEventRecord do some stream synchronization?
    SDAA_CALL(sdaaEventRecord(start_, SDAAThreadEntry::ThreadLocal()->stream));
  }
  virtual void Stop() { SDAA_CALL(sdaaEventRecord(stop_, SDAAThreadEntry::ThreadLocal()->stream)); }
  virtual int64_t SyncAndGetElapsedNanos() {
    SDAA_CALL(sdaaEventSynchronize(stop_));
    float milliseconds = 0;
    SDAA_CALL(sdaaEventElapsedTime(&milliseconds, start_, stop_));
    return milliseconds * 1e6;
  }
  virtual ~CUDATimerNode() {
    SDAA_CALL(sdaaEventDestroy(start_));
    SDAA_CALL(sdaaEventDestroy(stop_));
  }
  CUDATimerNode() {
    SDAA_CALL(sdaaEventCreate(&start_));
    SDAA_CALL(sdaaEventCreate(&stop_));
  }

  static constexpr const char* _type_key = "CUDATimerNode";
  TVM_DECLARE_FINAL_OBJECT_INFO(CUDATimerNode, TimerNode);

 private:
  sdaaEvent_t start_;
  sdaaEvent_t stop_;
};

TVM_REGISTER_OBJECT_TYPE(CUDATimerNode);

TVM_REGISTER_GLOBAL("profiling.timer.cuda").set_body_typed([](Device dev) {
  return Timer(make_object<CUDATimerNode>());
});

TVM_DLL String GetCudaFreeMemory() {
  size_t free_mem, total_mem;
  SDAA_CALL(sdaaMemGetInfo(&free_mem, &total_mem));
  std::stringstream ss;
  ss << "Current CUDA memory is " << free_mem << " bytes free out of " << total_mem
     << " bytes on device";
  return ss.str();
}

TVM_REGISTER_GLOBAL("runtime.GetCudaFreeMemory").set_body_typed(GetCudaFreeMemory);

}  // namespace runtime
}  // namespace tvm
