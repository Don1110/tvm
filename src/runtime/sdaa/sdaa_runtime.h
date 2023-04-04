#ifndef INCLUDE_SDAA_SDAA_RUNTIME_H_
#define INCLUDE_SDAA_SDAA_RUNTIME_H_

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "teco_detail/host_defines.h"
#include "teco_detail/teco_sdaa_vector_types.h"


/** Automatically select between Spin and Yield.*/
#define sdaaDeviceScheduleAuto 0x0

/** Dedicate a CPU core to spin-wait. Provides lowest latency, but burns a CPU core and may
 * consume more power.*/
#define sdaaDeviceScheduleSpin 0x1

#define SDAA_LAUNCH_PARAM_BUFFER_POINTER ((void*)0x01)
#define SDAA_LAUNCH_PARAM_BUFFER_SIZE ((void*)0x02)
#define SDAA_LAUNCH_PARAM_END ((void*)0x00)

#ifdef __cplusplus
#define __dparm(x) = x
#else
#define __dparm(x)
#endif  //__cplusplus

// Structure definitions:
#ifdef __cplusplus
extern "C" {
#endif  //__cplusplus


// Flags that can be used with hipEventCreateWithFlags.
/** Default flags.*/
#define sdaaEventDefault 0x0

/** Waiting will yield CPU. Power-friendly and usage-friendly but may increase latency.*/
#define sdaaEventBlockingSync 0x1

/** Disable event's capability to record timing information. May improve performance.*/
#define sdaaEventDisableTiming 0x2

/** Event can support IPC. Warnig: It is not supported in HIP.*/
#define sdaaEventInterprocess 0x4

typedef struct sdaaDeviceProp_t {
  char name[256];        /**< ASCII string identifying device */
  size_t totalGlobalMem; /**< total memory on device in bytes */
  int chipsetID;         /**< chip set id */
  int pciDeviceID;       /**< PCI device ID of the device */
  int pciVendor;         /**< PCI vendor ID of the device */
  int pciBusID;          /**< PCI bus ID of the device */
  int clockRate;         /**< Clock frequency in kilohertz */
} sdaaDeviceProp_t;

typedef sdaaDeviceProp_t sdaaDeviceProp;


/**
 * Memory type (for pointer attributes)
 */
typedef enum sdaaMemoryType {
  sdaaMemoryTypeUnregistered = 0,
  sdaaMemoryTypeHost,    ///< Memory is physically located on host
  sdaaMemoryTypeDevice,  ///< Memory is physically located on device. (see deviceId for specific
} sdaaMemoryType;


/**
 * Pointer attributes
 */
typedef struct sdaaPointerAttributes {
  enum sdaaMemoryType type;
  int device;
  void* devicePointer;
  void* hostPointer;
} sdaaPointerAttributes;

/*
 * @brief sdaaDeviceAttribute_t
 * @enum
 * @ingroup Enumerations
 */
typedef enum sdaaDeviceAttribute_t {
  sdaaDevAttrChipset = 0,
  sdaaDevAttrPciVendor,
  sdaaDevAttrPciBusId,
  sdaaDevAttrPciDeviceId,
  // Extended attributes for vendors
} sdaaDeviceAttribute_t;

typedef int sdaaDevice_t;

typedef struct isdaaStream_t* sdaaStream_t;

typedef struct isdaaEvent_t* sdaaEvent_t;

typedef struct isdaaModule_t* sdaaModule_t;

typedef struct isdaaModuleSymbol_t* sdaaFunction_t;

#define SDAA_IPC_HANDLE_SIZE 64
typedef struct sdaaIpcEventHandle_st {
  char reserved[SDAA_IPC_HANDLE_SIZE];
} sdaaIpcEventHandle_t;

typedef enum sdaaMemcpyKind {
  sdaaMemcpyHostToHost = 0,      ///< Host-to-Host Copy
  sdaaMemcpyHostToDevice = 1,    ///< Host-to-Device Copy
  sdaaMemcpyDeviceToHost = 2,    ///< Device-to-Host Copy
  sdaaMemcpyDeviceToDevice = 3,  ///< Device-to-Device Copy
  sdaaMemcpyDefault = 4          ///< Runtime will automatically determine copy-kind
                                 ///< based on virtual addresses.
} sdaaMemcpyKind;

enum sdaaStreamCaptureMode {
  sdaaStreamCaptureModeGlobal = 0,
  sdaaStreamCaptureModeThreadLocal,
  sdaaStreamCaptureModeRelaxed
};

enum sdaaStreamCaptureStatus {
  sdaaStreamCaptureStatusNone = 0,    ///< Stream is not capturing
  sdaaStreamCaptureStatusActive,      ///< Stream is actively capturing
  sdaaStreamCaptureStatusInvalidated  ///< Stream is part of a capture sequence
                                      ///< that has been invalidated, but not
                                      ///< terminated
};

//! Flags that can be used with hipStreamCreateWithFlags
/** Default stream creation flags. These are used with hipStreamCreate().*/
#define sdaaStreamDefault 0x00

/** Stream does not implicitly synchronize with null stream.*/
#define sdaaStreamNonBlocking 0x01

// Developer note - when updating these, update the sdaaErrorName and
// sdaaErrorString functions in NVCC and HCC paths Also update the
// sdaaCUDAErrorTosdaaError function in NVCC path.

typedef enum sdaaError_t {
  sdaaSuccess = 0,            ///< Successful completion.
  sdaaErrorInvalidValue = 1,  ///< One or more of the parameters passed to the API call is NULL
                              ///< or not in an acceptable range.
  sdaaErrorOutOfMemory = 2,
  // Deprecated
  sdaaErrorMemoryAllocation = 2,  ///< Memory allocation error.
  sdaaErrorNotInitialized = 3,
  // Deprecated
  sdaaErrorInitializationError = 3,
  sdaaErrorDeinitialized = 4,
  sdaaErrorsdaartUnloading = 4,
  sdaaErrorProfilerDisabled = 5,
  sdaaErrorProfilerNotInitialized = 6,
  sdaaErrorProfilerAlreadyStarted = 7,
  sdaaErrorProfilerAlreadyStopped = 8,
  sdaaErrorInvalidConfiguration = 9,
  sdaaErrorInvalidPitchValue = 12,
  sdaaErrorInvalidSymbol = 13,
  sdaaErrorInvalidDevicePointer = 17,    ///< Invalid Device Pointer
  sdaaErrorInvalidMemcpyDirection = 21,  ///< Invalid memory copy direction
  sdaaErrorInsufficientDriver = 35,
  sdaaErrorMissingConfiguration = 52,
  sdaaErrorPriorLaunchFailure = 53,
  sdaaErrorInvalidDeviceFunction = 98,
  sdaaErrorNoDevice = 100,       ///< Call to sdaaGetDeviceCount returned 0 devices
  sdaaErrorInvalidDevice = 101,  ///< DeviceID must be in range 0...#compute-devices.
  sdaaErrorInvalidImage = 200,
  sdaaErrorInvalidContext = 201,  ///< Produced when input context is invalid.
  sdaaErrorContextAlreadyCurrent = 202,
  sdaaErrorMapFailed = 205,
  // Deprecated
  sdaaErrorMapBufferObjectFailed = 205,  ///< Produced when the IPC memory attach failed from ROCr.
  sdaaErrorUnmapFailed = 206,
  sdaaErrorArrayIsMapped = 207,
  sdaaErrorAlreadyMapped = 208,
  sdaaErrorNoBinaryForGpu = 209,
  sdaaErrorAlreadyAcquired = 210,
  sdaaErrorNotMapped = 211,
  sdaaErrorNotMappedAsArray = 212,
  sdaaErrorNotMappedAsPointer = 213,
  sdaaErrorECCNotCorrectable = 214,
  sdaaErrorUnsupportedLimit = 215,
  sdaaErrorContextAlreadyInUse = 216,
  sdaaErrorPeerAccessUnsupported = 217,
  sdaaErrorInvalidKernelFile = 218,  ///< In CUDA DRV, it is CUDA_ERROR_INVALID_PTX
  sdaaErrorInvalidGraphicsContext = 219,
  sdaaErrorInvalidSource = 300,
  sdaaErrorFileNotFound = 301,
  sdaaErrorSharedObjectSymbolNotFound = 302,
  sdaaErrorSharedObjectInitFailed = 303,
  sdaaErrorOperatingSystem = 304,
  sdaaErrorInvalidHandle = 400,
  // Deprecated
  sdaaErrorInvalidResourceHandle = 400,  ///< Resource handle (sdaaEvent_t or sdaaStream_t) invalid.
  sdaaErrorNotFound = 500,
  sdaaErrorNotReady = 600,  ///< Indicates that asynchronous operations enqueued earlier are not
                            ///< ready.  This is not actually an error, but is used to
                            ///< distinguish from sdaaSuccess (which indicates completion). APIs
                            ///< that return this error include sdaaEventQuery and
                            ///< sdaaStreamQuery.
  sdaaErrorIllegalAddress = 700,
  sdaaErrorLaunchOutOfResources = 701,  ///< Out of resources error.
  sdaaErrorLaunchTimeOut = 702,
  sdaaErrorPeerAccessAlreadyEnabled =
      704,  ///< Peer access was already enabled from the current device.
  sdaaErrorPeerAccessNotEnabled = 705,  ///< Peer access was never enabled from the current device.
  sdaaErrorSetOnActiveProcess = 708,
  sdaaErrorContextIsDestroyed = 709,
  sdaaErrorAssert = 710,  ///< Produced when the kernel calls assert.
  sdaaErrorHostMemoryAlreadyRegistered =
      712,  ///< Produced when trying to lock a page-locked memory.
  sdaaErrorHostMemoryNotRegistered =
      713,                       ///< Produced when trying to unlock a non-page-locked memory.
  sdaaErrorLaunchFailure = 719,  ///< An exception occurred on the device while executing a kernel.
  sdaaErrorCooperativeLaunchTooLarge = 720,  ///< This error indicates that the number of blocks
                                             ///< launched per grid for a kernel that was launched
                                             ///< via cooperative launch APIs exceeds the maximum
                                             ///< number of allowed blocks for the current device
  sdaaErrorNotSupported = 801,  ///< Produced when the sdaa API is not supported/implemented
  sdaaErrorStreamCaptureUnsupported = 900,  ///< The operation is not permitted
                                            ///< when the stream is capturing.
  sdaaErrorStreamCaptureInvalidated = 901,  ///< The current capture sequence on the stream
                                            ///< has been invalidated due to a previous error.
  sdaaErrorStreamCaptureMerge = 902,        ///< The operation would have resulted in a merge of
                                            ///< two independent capture sequences.
  sdaaErrorStreamCaptureUnmatched = 903,    ///< The capture was not initiated in this stream.
  sdaaErrorStreamCaptureUnjoined = 904,     ///< The capture sequence contains a fork that was not
                                            ///< joined to the primary stream.
  sdaaErrorStreamCaptureIsolation = 905,    ///< A dependency would have been created which crosses
                                            ///< the capture sequence boundary. Only implicit
                                            ///< in-stream ordering dependencies  are allowed
                                            ///< to cross the boundary
  sdaaErrorStreamCaptureImplicit = 906,     ///< The operation would have resulted in a disallowed
                                            ///< implicit dependency on a current capture sequence
                                            ///< from sdaaStreamLegacy.
  sdaaErrorCapturedEvent = 907,  ///< The operation is not permitted on an event which was last
                                 ///< recorded in a capturing stream.
  sdaaErrorStreamCaptureWrongThread = 908,  ///< A stream capture sequence not initiated with
                                            ///< the sdaaStreamCaptureModeRelaxed argument to
                                            ///< sdaaStreamBeginCapture was passed to
                                            ///< sdaaStreamEndCapture in a different thread.
  sdaaErrorUnknown = 999,                   //< Unknown error.
  // HSA Runtime Error Codes start here.
  sdaaErrorRuntimeMemory = 1052,  ///< HSA runtime memory call returned error.
                                  ///< Typically not seen in production systems.
  sdaaErrorRuntimeOther = 1053,   ///< HSA runtime call other than memory returned error.  Typically
                                  ///< not seen in production systems.
  sdaaErrorTbd                    ///< Marker that more error codes are needed.
} sdaaError_t;

// Stream per thread
/** Implicit stream per application thread.*/
#define sdaaStreamPerThread ((sdaaStream_t)2)

/**
 * Struct for data in 3D
 *
 */
typedef struct dim3 {
  uint32_t x;  ///< x
  uint32_t y;  ///< y
  uint32_t z;  ///< z
#ifdef __cplusplus
  constexpr __host__ __device__ dim3(uint32_t _x = 1, uint32_t _y = 1, uint32_t _z = 1)
      : x(_x), y(_y), z(_z) {}
#endif  //__cplusplus
} dim3;


enum sdaaLimit {
  // hipLimitStackSize = 0x0,        // limit device stack size
  sdaaLimitPrintfFifoSize = 0x01,  // limit printf fifo size
  // hipLimitMallocHeapSize = 0x02,  // limit heap size
  sdaaLimitRange  // supported limit range
};

/*******************************************************************************
 *                                                                              *
 *            Device Management *
 *                                                                              *
 *******************************************************************************/
sdaaError_t sdaaDeviceReset(void);

sdaaError_t sdaaSetDevice(int device);

sdaaError_t sdaaGetDevice(int* device);

sdaaError_t sdaaGetDeviceCount(int* count);

sdaaError_t sdaaDeviceSynchronize(void);

sdaaError_t sdaaGetDeviceProperties(sdaaDeviceProp_t* props, int device);

sdaaError_t sdaaDeviceGetAttribute(int* value, sdaaDeviceAttribute_t attr, int device);

sdaaError_t sdaaDeviceSetLimit(enum sdaaLimit limit, size_t value);

sdaaError_t sdaaDeviceGetLimit(size_t* pValue, enum sdaaLimit limit);
/*******************************************************************************
 *                                                                              *
 *               Error Handling *
 *                                                                              *
 *******************************************************************************/
const char* sdaaGetErrorName(sdaaError_t error);
const char* sdaaGetErrorString(sdaaError_t error);

sdaaError_t sdaaGetLastError(void);

sdaaError_t sdaaPeekAtLastError(void);

/*******************************************************************************
 *                                                                              *
 *               Stream Management *
 *                                                                              *
 *******************************************************************************/
sdaaError_t sdaaStreamCreate(sdaaStream_t* stream);

sdaaError_t sdaaStreamCreateWithFlags(sdaaStream_t* pStream, unsigned int flags);

sdaaError_t sdaaStreamDestroy(sdaaStream_t stream);

sdaaError_t sdaaStreamSynchronize(sdaaStream_t stream __dparm(0));

sdaaError_t sdaaStreamQuery(sdaaStream_t stream);

sdaaError_t sdaaStreamGetFlags(sdaaStream_t stream, unsigned int* flags);

sdaaError_t sdaaStreamWaitEvent(sdaaStream_t stream, sdaaEvent_t event, unsigned int flags);

typedef void (*sdaaStreamCallback_t)(sdaaStream_t stream, sdaaError_t status, void* userData);
sdaaError_t sdaaStreamAddCallback(sdaaStream_t stream, sdaaStreamCallback_t callback,
                                  void* userData, unsigned int flags);

/*******************************************************************************
 *                                                                              *
 *               Event Management *
 *                                                                              *
 *******************************************************************************/

sdaaError_t sdaaEventCreate(sdaaEvent_t* event);

sdaaError_t sdaaEventRecord(sdaaEvent_t event, sdaaStream_t stream __dparm(0));

sdaaError_t sdaaEventDestroy(sdaaEvent_t event);

sdaaError_t sdaaEventQuery(sdaaEvent_t event);

sdaaError_t sdaaEventSynchronize(sdaaEvent_t event);

sdaaError_t sdaaEventElapsedTime(float* ms, sdaaEvent_t start, sdaaEvent_t end);

sdaaError_t sdaaEventCreateWithFlags(sdaaEvent_t* event, unsigned flags);

sdaaError_t sdaaIpcGetEventHandle(sdaaIpcEventHandle_t* handle, sdaaEvent_t event);

sdaaError_t sdaaIpcOpenEventHandle(sdaaEvent_t* event, sdaaIpcEventHandle_t handle);


// *******************************************************************************
//  *                                                                              *
//  *               Execution Control *
//  *                                                                              *
// *******************************************************************************/
sdaaError_t sdaaModuleLoad(sdaaModule_t* module, const char* fname);

sdaaError_t sdaaModuleUnload(sdaaModule_t module);
sdaaError_t sdaaModuleGetFunction(sdaaFunction_t* hfunc, sdaaModule_t hmod, const char* name);
sdaaError_t sdaaModuleLaunchKernel(sdaaFunction_t f, int core_num, sdaaStream_t hStream,
                                   void** kernelParams);

/* for compiler */
void** __sdaaRegisterFatBinary(const void* data);
void __sdaaUnregisterFatBinary(void** module);
void __sdaaRegisterFunction(void** modules, const void* hostFunction, char* deviceFunction,
                            const char* deviceName);

sdaaError_t __sdaaPushCallConfiguration(size_t sharedMem, sdaaStream_t stream __dparm(0));
sdaaError_t __sdaaPopCallConfiguration(size_t* sharedMem, sdaaStream_t* stream);
sdaaError_t sdaaSetupArgument(const void* arg, size_t size, size_t offset);
sdaaError_t sdaaLaunchKernel(const void* function_address, void** args,
                             size_t sharedMemBytes __dparm(0), sdaaStream_t stream __dparm(0));
// *******************************************************************************
//  *                                                                              *
//  *               Memory Management *
//  *                                                                              *
// *******************************************************************************/

sdaaError_t sdaaMalloc(void** ptr, size_t size);

sdaaError_t sdaaMallocPitch(void** ptr, size_t* pitch, size_t width, size_t height);

sdaaError_t sdaaMallocHost(void** ptr, size_t size);

sdaaError_t sdaaMallocCross(void** ptr, size_t size);

sdaaError_t sdaaFree(void* ptr);

sdaaError_t sdaaFreeHost(void* ptr);

sdaaError_t sdaaMemcpy(void* dst, const void* src, size_t sizeBytes, sdaaMemcpyKind kind);

sdaaError_t sdaaMemcpyAsync(void* dst, const void* src, size_t sizeBytes, sdaaMemcpyKind kind,
                            sdaaStream_t stream __dparm(0));
sdaaError_t sdaaMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                         size_t height, enum sdaaMemcpyKind kind);

sdaaError_t sdaaMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch,
                              size_t width, size_t height, enum sdaaMemcpyKind kind,
                              sdaaStream_t stream);

sdaaError_t sdaaMemset(void* dst, int value, size_t sizeBytes);

sdaaError_t sdaaMemsetAsync(void* dst, int value, size_t sizeBytes, sdaaStream_t stream __dparm(0));

sdaaError_t sdaaMemsetD8(void* dst, unsigned char value, size_t count);

sdaaError_t sdaaMemsetD16(void* dst, unsigned short value, size_t count);

sdaaError_t sdaaMemsetD32(void* dst, unsigned int value, size_t count);

sdaaError_t sdaaMemGetInfo(size_t* free, size_t* total);

sdaaError_t sdaaPointerGetAttributes(sdaaPointerAttributes* attributes, const void* ptr);

// *******************************************************************************
//  *                                                                              *
//  *               Version Management *
//  *                                                                              *
// *******************************************************************************/

sdaaError_t sdaaRuntimeGetVersion(int* runtimeVersion);

sdaaError_t sdaaDriverGetVersion(int* driverVersion);

#define KNRM "\x1B[0m"
#define KRED "\x1B[31m"
#define KGRN "\x1B[32m"
#define KYEL "\x1B[33m"
#define KBLU "\x1B[34m"
#define KMAG "\x1B[35m"
#define KCYN "\x1B[36m"
#define KWHT "\x1B[37m"

#define SDAACHECK(error)                                                                           \
  {                                                                                                \
    sdaaError_t localError = error;                                                                \
    if ((localError != sdaaSuccess) && (localError != sdaaErrorPeerAccessAlreadyEnabled)) {        \
      printf("%serror: '%s'(%d) from %s at %s:%d%s\n", KRED, sdaaGetErrorString(localError),       \
             localError, #error, __FUNCTION__, __LINE__, KNRM);                                    \
      exit(EXIT_FAILURE);                                                                          \
    }                                                                                              \
  }

// 暂时向老接口兼容，checkSdaaErrors 后期会被废止
#define checkSdaaErrors SDAACHECK


// Structure definitions:
#ifdef __cplusplus
}
#endif

// /**
//  * @brief: C++ wrapper for hipMalloc
//  *
//  * Perform automatic type conversion to eliminate need for excessive typecasting (ie void**)
//  *
//  * __HIP_DISABLE_CPP_FUNCTIONS__ macro can be defined to suppress these
//  * wrappers. It is useful for applications which need to obtain decltypes of
//  * HIP runtime APIs.
//  *
//  * @see sdaaMalloc
//  */
#if defined(__cplusplus)
template <class T> static inline sdaaError_t sdaaMalloc(T** devPtr, size_t size) {
  return sdaaMalloc((void**)devPtr, size);
}

// Provide an override to automatically typecast the pointer type from void**, and also provide a
// default for the flags.
template <class T> static inline sdaaError_t sdaaMallocHost(T** ptr, size_t size) {
  return sdaaMallocHost((void**)ptr, size);
}

#endif  //__cplusplus
#endif  // INCLUDE_SDAA_SDAA_RUNTIME_H_
