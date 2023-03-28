// /*! \file sdaa_runtime.h
// 	\author Andrew Kerr <arkerr@gatech.edu>
// 	\brief implements an up-to-date SDAA Runtime API
// 	\date 11 Dec 2009
// */

#ifndef SDAA_RUNTIME_H_INCLUDED
#define SDAA_RUNTIME_H_INCLUDED

// C includes
#include <string.h>
#include <limits.h>
#include <stdint.h>

// Ocelot includes
//#include <ocelot/sdaa/interface/sdaaFatBinary.h>
#include "sdaaFatBinary.h"


#ifdef __cplusplus
extern "C" {
#endif


typedef int sdaaEvent_t;
typedef int sdaaStream_t;
//typedef unsigned int GLuint;

#define sdaaHostAllocDefault        0   ///< Default page-locked allocation flag
#define sdaaHostAllocPortable       1   ///< Pinned memory accessible by all SDAA contexts
#define sdaaHostAllocMapped         2   ///< Map allocation into device space
#define sdaaHostAllocWriteCombined  4   ///< Write-combined memory

#define sdaaEventDefault            0   ///< Default event flag
#define sdaaEventBlockingSync       1   ///< Event uses blocking synchronization

#define sdaaDeviceScheduleAuto      0   ///< Device flag - Automatic scheduling
#define sdaaDeviceScheduleSpin      1   ///< Device flag - Spin default scheduling
#define sdaaDeviceScheduleYield     2   ///< Device flag - Yield default scheduling
#define sdaaDeviceBlockingSync      4   ///< Device flag - Use blocking synchronization
#define sdaaDeviceMapHost           8   ///< Device flag - Support mapped pinned allocations
#define sdaaDeviceLmemResizeToMax   16  ///< Device flag - Keep local memory allocation after launch
#define sdaaDeviceMask              0x1f ///< Device flags mask

enum sdaaMemcpyKind {
	sdaaMemcpyHostToHost = 0,
	sdaaMemcpyHostToDevice = 1,
	sdaaMemcpyDeviceToHost = 2,
	sdaaMemcpyDeviceToDevice = 3
};

enum sdaaChannelFormatKind {
	sdaaChannelFormatKindSigned = 0,
	sdaaChannelFormatKindUnsigned = 1,
	sdaaChannelFormatKindFloat = 2,
	sdaaChannelFormatKindNone = 3
};

enum sdaaComputeMode {
	sdaaComputeModeDefault,
	sdaaComputeModeExclusive,
	sdaaComputeModeProhibited
};

enum sdaaError
{
  sdaaSuccess                           =      0,   ///< No errors
  sdaaErrorMissingConfiguration         =      1,   ///< Missing configuration error
  sdaaErrorMemoryAllocation             =      2,   ///< Memory allocation error
  sdaaErrorInitializationError          =      3,   ///< Initialization error
  sdaaErrorLaunchFailure                =      4,   ///< Launch failure
  sdaaErrorPriorLaunchFailure           =      5,   ///< Prior launch failure
  sdaaErrorLaunchTimeout                =      6,   ///< Launch timeout error
  sdaaErrorLaunchOutOfResources         =      7,   ///< Launch out of resources error
  sdaaErrorInvalidDeviceFunction        =      8,   ///< Invalid device function
  sdaaErrorInvalidConfiguration         =      9,   ///< Invalid configuration
  sdaaErrorInvalidDevice                =     10,   ///< Invalid device
  sdaaErrorInvalidValue                 =     11,   ///< Invalid value
  sdaaErrorInvalidPitchValue            =     12,   ///< Invalid pitch value
  sdaaErrorInvalidSymbol                =     13,   ///< Invalid symbol
  sdaaErrorMapBufferObjectFailed        =     14,   ///< Map buffer object failed
  sdaaErrorUnmapBufferObjectFailed      =     15,   ///< Unmap buffer object failed
  sdaaErrorInvalidHostPointer           =     16,   ///< Invalid host pointer
  sdaaErrorInvalidDevicePointer         =     17,   ///< Invalid device pointer
  sdaaErrorInvalidChannelDescriptor     =     20,   ///< Invalid channel descriptor
  sdaaErrorInvalidMemcpyDirection       =     21,   ///< Invalid memcpy direction
  sdaaErrorAddressOfConstant            =     22,   ///< Address of constant error
  sdaaErrorSynchronizationError         =     25,   ///< Synchronization error
  sdaaErrorInvalidFilterSetting         =     26,   ///< Invalid filter setting
  sdaaErrorInvalidNormSetting           =     27,   ///< Invalid norm setting
  sdaaErrorMixedDeviceExecution         =     28,   ///< Mixed device execution
  sdaaErrorSdaartUnloading              =     29,   ///< SDAA runtime unloading
  sdaaErrorUnknown                      =     30,   ///< Unknown error condition
  sdaaErrorNotYetImplemented            =     31,   ///< Function not yet implemented
  sdaaErrorMemoryValueTooLarge          =     32,   ///< Memory value too large
  sdaaErrorInvalidResourceHandle        =     33,   ///< Invalid resource handle
  sdaaErrorNotReady                     =     34,   ///< Not ready error
  sdaaErrorInsufficientDriver           =     35,   ///< SDAA runtime is newer than driver
  sdaaErrorSetOnActiveProcess           =     36,   ///< Set on active process error
  sdaaErrorNoDevice                     =     38,   ///< No available SDAA device
  sdaaErrorECCUncorrectable             =     39,   ///< Uncorrectable ECC error detected
  sdaaErrorStartupFailure               =   0x7f,   ///< Startup failure
  sdaaErrorApiFailureBase               =  10000    ///< API failure base
};

typedef enum sdaaError sdaaError_t;

enum sdaaLimit
{
    sdaaLimitStackSize      = 0x00, //< GPU thread stack size
    sdaaLimitPrintfFifoSize = 0x01, //< GPU printf FIFO size
    sdaaLimitMallocHeapSize = 0x02  //< GPU malloc heap size
};


#define sdaaDevIdle 0x0
#define sdaaDevBusy 0x1
#define sdaaDevError 0x2
typedef int sdaaDevState;

#define sdaaDevCgNoFault 0x0
#define sdaaDevCgPmFault 0x1
#define sdaaDevCgSpeFault 0x2
#define sdaaDevCgMpeFault 0x3

typedef int sdaaDevFaultType;

struct uint3 {
	unsigned int x, y, z;
};

struct dim3
{
    unsigned int x, y, z;
#if defined(__cplusplus) && !defined(__SDAABE__)
    dim3(unsigned int x = 1, unsigned int y = 1, unsigned int z = 1) : x(x), y(y), z(z) {}
    dim3(uint3 v) : x(v.x), y(v.y), z(v.z) {}
    operator uint3(void) { uint3 t; t.x = x; t.y = y; t.z = z; return t; }
#endif
};

/*DEVICE_BUILTIN*/
typedef struct dim3 dim3;

struct sdaaExtent {
	size_t width;
	size_t height;
	size_t depth;
};

struct sdaaDeviceProp
{
    char   name[256];                  /**< ASCII string identifying device */
    size_t totalGlobalMem;             /**< Global memory available on device in bytes */
    size_t memPitch;                   /**< Maximum pitch in bytes allowed by memory copies */
    int    maxThreadsPerBlock;         /**< Maximum number of threads per block */
    int    maxThreadsDim[3];           /**< Maximum size of each dimension of a block */
    int    maxGridSize[3];             /**< Maximum size of each dimension of a grid */
    int    clockRate;                  /**< Clock frequency in kilohertz */
    size_t totalConstMem;              /**< Constant memory available on device in bytes */
    int    major;                      /**< Major compute capability */
    int    minor;                      /**< Minor compute capability */
    int    deviceOverlap;              /**< Device can concurrently copy memory and execute a kernel. Deprecated. Use instead asyncEngineCount. */
    int    multiProcessorCount;        /**< Number of multiprocessors on device */
    int    kernelExecTimeoutEnabled;   /**< Specified whether there is a run time limit on kernels */
    int    integrated;                 /**< Device is integrated as opposed to discrete */
    int    canMapHostMemory;           /**< Device can map host memory with sdaaHostAlloc/sdaaHostGetDevicePointer */
    int    computeMode;                /**< Compute mode (See ::sdaaComputeMode) */
    int    concurrentKernels;          /**< Device can possibly execute multiple kernels concurrently */
    int    ECCEnabled;                 /**< Device has ECC support enabled */
    int    pciBusID;                   /**< PCI bus ID of the device */
    int    pciDeviceID;                /**< PCI device ID of the device */
    int    pciDomainID;                /**< PCI domain ID of the device */
    int    tccDriver;                  /**< 1 if device is a Tesla device using TCC driver, 0 otherwise */
    int    asyncEngineCount;           /**< Number of asynchronous engines */
    int    unifiedAddressing;          /**< Device shares a unified address space with the host */
    int    memoryClockRate;            /**< Peak memory clock frequency in kilohertz */
    int    memoryBusWidth;             /**< Global memory bus width in bits */
    int    maxThreadsPerMultiProcessor;/**< Maximum resident threads per multiprocessor */
};

/**
 * SDAA device attributes
 */
enum sdaaDeviceAttr
{
    sdaaDevAttrMaxPitch                       = 11, /**< Maximum pitch in bytes allowed by memory copies */
    sdaaDevAttrClockRate                      = 13, /**< Peak clock frequency in kilohertz */
    sdaaDevAttrGpuOverlap                     = 15, /**< Device can possibly copy memory and execute a kernel concurrently */
    sdaaDevAttrKernelExecTimeout              = 17, /**< Specifies whether there is a run time limit on kernels */
    sdaaDevAttrIntegrated                     = 18, /**< Device is integrated with host memory */
    sdaaDevAttrComputeMode                    = 20, /**< Compute mode (See ::sdaaComputeMode for details) */
    sdaaDevAttrSurfaceAlignment               = 30, /**< Alignment requirement for surfaces */
    sdaaDevAttrConcurrentKernels              = 31, /**< Device can possibly execute multiple kernels concurrently */
    sdaaDevAttrEccEnabled                     = 32, /**< Device has ECC support enabled */
    sdaaDevAttrPciBusId                       = 33, /**< PCI bus ID of the device */
    sdaaDevAttrPciDeviceId                    = 34, /**< PCI device ID of the device */
    sdaaDevAttrTccDriver                      = 35, /**< Device is using TCC driver model */
    sdaaDevAttrMemoryClockRate                = 36, /**< Peak memory clock frequency in kilohertz */
    sdaaDevAttrGlobalMemoryBusWidth           = 37, /**< Global memory bus width in bits */
    sdaaDevAttrL2CacheSize                    = 38, /**< Size of L2 cache in bytes */
    sdaaDevAttrMaxThreadsPerMultiProcessor    = 39, /**< Maximum resident threads per multiprocessor */
    sdaaDevAttrAsyncEngineCount               = 40, /**< Number of asynchronous engines */
    sdaaDevAttrUnifiedAddressing              = 41, /**< Device shares a unified address space with the host */    
    sdaaDevAttrPciDomainId                    = 50, /**< PCI domain ID of the device */
    sdaaDevAttrMaxSurface1DWidth              = 55, /**< Maximum 1D surface width */
    sdaaDevAttrMaxSurface1DLayeredWidth       = 61, /**< Maximum 1D layered surface width */
    sdaaDevAttrMaxSurface1DLayeredLayers      = 62, /**< Maximum layers in a 1D layered surface */
    sdaaDevAttrMaxSurfaceCubemapWidth         = 66, /**< Maximum cubemap surface width */
    sdaaDevAttrMaxSurfaceCubemapLayeredWidth  = 67, /**< Maximum cubemap layered surface width */
    sdaaDevAttrMaxSurfaceCubemapLayeredLayers = 68, /**< Maximum layers in a cubemap layered surface */
    sdaaDevAttrComputeCapabilityMajor         = 75, /**< Major compute capability version number */ 
    sdaaDevAttrComputeCapabilityMinor         = 76, /**< Minor compute capability version number */
};

struct sdaaFuncAttributes {
   size_t sharedSizeBytes;  ///< Size of shared memory in bytes
   size_t constSizeBytes;   ///< Size of constant memory in bytes
   size_t localSizeBytes;   ///< Size of local memory in bytes
   int maxThreadsPerBlock;  ///< Maximum number of threads per block
   int numRegs;             ///< Number of registers used
   int ptxVersion;          ///< PTX version number eq 21
   int binaryVersion;       ///< binary version 
};

struct sdaaPitchedPtr {
	void *ptr;
	size_t pitch;
	size_t xsize;
	size_t ysize;
};

struct sdaaPos {
	size_t x;
	size_t y;
	size_t z;
};


typedef struct SDuuid_st sdaaUUID_t;

/*
 * Function        : Select a load image from the __sdaaFat binary
 *                   that will run on the specified GPU.
 * Parameters      : binary  (I) Fat binary
 *                   policy  (I) Parameter influencing the selection process in case no
 *                               fully matching cubin can be found, but instead a choice can
 *                               be made between ptx compilation or selection of a
 *                               cubin for a less capable GPU.
 *                   gpuName (I) Name of target GPU
 *                   cubin   (O) Returned cubin text string, or NULL when 
 *                               no matching cubin for the specified gpu
 *                               could be found.
 *                   dbgInfo (O) If this parameter is not NULL upon entry, then
 *                               the name of a file containing debug information
 *                               on the returned cubin will be returned, or NULL 
 *                               will be returned when cubin or such debug info 
 *                               cannot be found.
 */
void fatGetCubinForGpuWithPolicy( __sdaaFatSdaaBinary *binary, __sdaaFatCompilationPolicy policy, char* gpuName, char* *cubin, char* *dbgInfoFile );

#define fatGetCubinForGpu(binary,gpuName,cubin,dbgInfoFile) \
          fatGetCubinForGpuWithPolicy(binary,__sdaaFatAvoidPTX,gpuName,cubin,dbgInfoFile)

/*
 * Function        : Check if a binary will be JITed for the specified target architecture
 * Parameters      : binary  (I) Fat binary
 *                   policy  (I) Compilation policy, as described by fatGetCubinForGpuWithPolicy
 *                   gpuName (I) Name of target GPU
 *                   ptx     (O) PTX string to be JITed
 * Function Result : True if the given binary will be JITed; otherwise, False
 */
unsigned char fatCheckJitForGpuWithPolicy( __sdaaFatSdaaBinary *binary, __sdaaFatCompilationPolicy policy, char* gpuName, char* *ptx );

#define fatCheckJitForGpu(binary,gpuName,ptx) \
          fatCheckJitForGpuWithPolicy(binary,__sdaaFatAvoidPTX,gpuName,ptx)

/*
 * Function        : Free information previously obtained via function fatGetCubinForGpu.
 * Parameters      : cubin   (I) Cubin text string to free
 *                   dbgInfo (I) Debug info filename to free, or NULL
 */
void fatFreeCubin( char* cubin, char* dbgInfoFile );


/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

extern void** __sdaaRegisterFatBinary(void *fatCubin);

extern void __sdaaUnregisterFatBinary(void **fatCubinHandle);


extern void __sdaaRegisterFunction( void **binHandle, char *deviceFun);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

extern sdaaError_t sdaaMemsetAsync(void *devPtr, int value, size_t count, sdaaStream_t stream);


/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

extern sdaaError_t sdaaMalloc(void **devPtr, size_t size);
extern sdaaError_t sdaaMallocHost(void **ptr, size_t size);
extern sdaaError_t sdaaFree(void *devPtr);
extern sdaaError_t sdaaFreeHost(void *ptr);

extern sdaaError_t sdaaPrintInfo(void *addr, unsigned long size, unsigned int type, sdaaStream_t stream);
//sdaa IPC memory handle api
extern sdaaError_t  sdaaMemGetP2PAddr(void **addr, void *p); 
extern sdaaError_t  sdaaMemGetPhysAddr(void **addr, void *p); 
extern sdaaError_t  sdaaIpcGetMemHandle(int *mem_handle, void *devPtr); 
extern sdaaError_t  sdaaIpcOpenMemHandle(void** devPtr, int handle, unsigned int flags);
extern sdaaError_t  sdaaIpcCloseMemHandle(void* devPtr);

extern sdaaError_t sdaaHostAlloc(void **pHost, size_t bytes, unsigned int flags);
extern sdaaError_t sdaaHostGetFlags(unsigned int *pFlags, void *pHost);

#if 0
extern sdaaError_t sdaaHostGetDevicePointer(void **pDevice, void *pHost, 
	unsigned int flags);
extern sdaaError_t sdaaHostRegister(void *pHost, size_t bytes, unsigned int flags);
extern sdaaError_t sdaaHostUnregister(void *pHost);
#endif

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

extern sdaaError_t sdaaMemcpy(void *dst, const void *src, size_t count, 
	enum sdaaMemcpyKind kind);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

extern sdaaError_t sdaaMemcpyAsync(void *dst, const void *src, size_t count, 
	enum sdaaMemcpyKind kind, sdaaStream_t stream);
/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

extern sdaaError_t sdaaMemset(void *devPtr, int value, size_t count);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if 0
extern sdaaError_t sdaaGetSymbolAddress(void **devPtr, const char *symbol);
extern sdaaError_t sdaaGetSymbolSize(size_t *size, const char *symbol);
#endif
/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

extern sdaaError_t sdaaGetDeviceCount(int *count);
extern sdaaError_t sdaaGetDeviceProperties(struct sdaaDeviceProp *prop, int device);
extern sdaaError_t sdaaChooseDevice(int *device, const struct sdaaDeviceProp *prop);
extern sdaaError_t sdaaSetDevice(int device);
extern sdaaError_t sdaaGetDevice(int *device);
extern sdaaError_t sdaaSetValidDevices(int *device_arr, int len);
extern sdaaError_t sdaaSetDeviceFlags( int flags );
extern sdaaError_t sdaaGetDeviceState(sdaaDevState *device_state, sdaaDevFaultType *fault_type, unsigned long *cg_fault_reg);
extern sdaaError_t sdaaDeviceGetAttribute( int* value, enum sdaaDeviceAttr attrbute,
	int device );



/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

extern sdaaError_t sdaaGetLastError(void);
extern sdaaError_t sdaaPeekAtLastError();
extern const char* sdaaGetErrorString(sdaaError_t error);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

//extern sdaaError_t sdaaConfigureCall(dim3 gridDim, dim3 blockDim, 
//	size_t sharedMem = 0, sdaaStream_t stream = 0);
extern sdaaError_t sdaaConfigureCall(int arrayNum);
extern sdaaError_t sdaaSetupArgument(const void *arg, size_t size, 
	size_t offset);
extern sdaaError_t sdaaLaunch(const char *entry);
extern sdaaError_t sdaaLaunchAsync(const char *entry, sdaaStream_t stream);
extern sdaaError_t sdaaFuncGetAttributes(struct sdaaFuncAttributes *attr, 
	const char *func);
extern sdaaError_t sdaaStreamWaitEvent(sdaaStream_t stream, sdaaEvent_t event, unsigned int flags);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

extern sdaaError_t sdaaStreamCreate(sdaaStream_t *pStream);
extern sdaaError_t sdaaStreamDestroy(sdaaStream_t stream);
extern sdaaError_t sdaaStreamSynchronize(sdaaStream_t stream);
extern sdaaError_t sdaaStreamQuery(sdaaStream_t stream);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

extern sdaaError_t sdaaEventCreate(sdaaEvent_t *event);
extern sdaaError_t sdaaEventCreateWithFlags(sdaaEvent_t *event, int flags);
extern sdaaError_t sdaaEventRecord(sdaaEvent_t event, sdaaStream_t stream);
extern sdaaError_t sdaaEventQuery(sdaaEvent_t event);
extern sdaaError_t sdaaEventSynchronize(sdaaEvent_t event);
extern sdaaError_t sdaaEventDestroy(sdaaEvent_t event);
extern sdaaError_t sdaaEventElapsedTime(float *ms, sdaaEvent_t start, sdaaEvent_t end);


/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/


/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/
#if 0
extern sdaaError_t sdaaSetDoubleForDevice(double *d);
extern sdaaError_t sdaaSetDoubleForHost(double *d);
#endif
/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

extern sdaaError_t sdaaDeviceReset(void);
extern sdaaError_t sdaaDeviceSynchronize(void);
extern sdaaError_t sdaaDeviceSetLimit(enum sdaaLimit limit, size_t value);
extern sdaaError_t sdaaDeviceGetLimit(size_t *pValue, enum sdaaLimit limit);

extern sdaaError_t sdaaThreadExit(void);
extern sdaaError_t sdaaThreadSynchronize(void);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

extern sdaaError_t sdaaDriverGetVersion(int *driverVersion);
extern sdaaError_t sdaaRuntimeGetVersion(int *runtimeVersion);

#define checkSdaaErrors( a ) do { \
    if (sdaaSuccess != (a)) { \
    fprintf(stderr, "Sdaa runtime error in line %d of file %s \
    : %s \n", __LINE__, __FILE__, sdaaGetErrorString(sdaaGetLastError()) ); \
    exit(EXIT_FAILURE); \
    } \
    } while(0);

#ifdef __cplusplus
}
#endif

#endif

