/*
 * Copyright (C) 2011 Shinpei Kato
 *
 * Systems Research Lab, University of California at Santa Cruz
 * All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef __SDAA_H__
#define __SDAA_H__

#ifdef __KERNEL__
#include <linux/types.h>
#else
#include <stddef.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

typedef unsigned long long SDdeviceptr;
typedef int SDdevice;
typedef struct SDctx_st* SDcontext;
typedef struct SDmod_st* SDmodule;
typedef struct SDfunc_st* SDfunction;
typedef struct SDevent_st* SDevent;
typedef struct SDstream_st* SDstream;

/**
 * Context creation flags
 */
typedef enum SDctx_flags_enum {
    SD_CTX_SCHED_AUTO          = 0x00, /**< Automatic scheduling */
    SD_CTX_SCHED_SPIN          = 0x01, /**< Set spin as default scheduling */
    SD_CTX_SCHED_YIELD         = 0x02, /**< Set yield as default scheduling */
    SD_CTX_SCHED_BLOCKING_SYNC = 0x04, /**< Set blocking synchronization as default scheduling */
    SD_CTX_BLOCKING_SYNC       = 0x04, /**< Set blocking synchronization as default scheduling \deprecated */
    SD_CTX_SCHED_MASK          = 0x07, 
    SD_CTX_MAP_HOST            = 0x08, /**< Support mapped pinned allocations */
    SD_CTX_LMEM_RESIZE_TO_MAX  = 0x10, /**< Keep local memory allocation after launch */
    SD_CTX_FLAGS_MASK          = 0x1f
} SDctx_flags;

/**
 * Event creation flags
 */
typedef enum SDevent_flags_enum {
    SD_EVENT_DEFAULT        = 0x0, /**< Default event flag */
    SD_EVENT_BLOCKING_SYNC  = 0x1, /**< Event uses blocking synchronization */
    SD_EVENT_DISABLE_TIMING = 0x2, /**< Event will not record timing data */
    SD_EVENT_INTERPROCESS   = 0x4  /**< Event is suitable for interprocess use. SD_EVENT_DISABLE_TIMING must be set */
} SDevent_flags;


/**
 * Device properties
 */
typedef enum SDdevice_attribute_enum {
    SD_DEVICE_ATTRIBUTE_MAX_PITCH = 11,                         /**< Maximum pitch in bytes allowed by memory copies */
    SD_DEVICE_ATTRIBUTE_CLOCK_RATE = 13,                        /**< Peak clock frequency in kilohertz */
    SD_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15,                       /**< Device can possibly copy memory and execute a kernel concurrently. Deprecated. Use instead SD_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT. */
    SD_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = 17,               /**< Specifies whether there is a run time limit on kernels */
    SD_DEVICE_ATTRIBUTE_INTEGRATED = 18,                        /**< Device is integrated with host memory */
    SD_DEVICE_ATTRIBUTE_COMPUTE_MODE = 20,                      /**< Compute mode (See ::SDcomputemode for details) */
    SD_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = 31,                /**< Device can possibly execute multiple kernels concurrently */
    SD_DEVICE_ATTRIBUTE_ECC_ENABLED = 32,                       /**< Device has ECC support enabled */
    SD_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33,                        /**< PCI bus ID of the device */
    SD_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34,                     /**< PCI device ID of the device */
    SD_DEVICE_ATTRIBUTE_TCC_DRIVER = 35,                        /**< Device is using TCC driver model */
    SD_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36,                 /**< Peak memory clock frequency in kilohertz */
    SD_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37,           /**< Global memory bus width in bits */
    SD_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39,    /**< Maximum resident threads per multiprocessor */
    SD_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40,                /**< Number of asynchronous engines */
    SD_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41,                /**< Device shares a unified address space with the host */    
    SD_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 50,                     /**< PCI domain ID of the device */
} SDdevice_attribute;

/**
 * Legacy device properties
 */
typedef struct SDdevprop_st {
    int memPitch;               /**< Maximum pitch in bytes allowed by memory copies */
    int clockRate;              /**< Clock frequency in kilohertz */
    int textureAlign;           /**< Alignment requirement for textures */
} SDdevprop;

/**
 * Pointer information
 */
typedef enum SDpointer_attribute_enum {
    SD_POINTER_ATTRIBUTE_CONTEXT = 1,        /**< The ::SDcontext on which a pointer was allocated or registered */
    SD_POINTER_ATTRIBUTE_MEMORY_TYPE = 2,    /**< The ::SDmemorytype describing the physical location of a pointer */
    SD_POINTER_ATTRIBUTE_DEVICE_POINTER = 3, /**< The address at which a pointer's memory may be accessed on the device */
    SD_POINTER_ATTRIBUTE_HOST_POINTER = 4,   /**< The address at which a pointer's memory may be accessed on the host */
} SDpointer_attribute;

/**
 * Function properties
 */
typedef enum SDfunction_attribute_enum {
    /**
     * The maximum number of threads per block, beyond which a launch of the
     * function would fail. This number depends on both the function and the
     * device on which the function is currently loaded.
     */
    SD_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0,

    /**
     * The size in bytes of statically-allocated shared memory required by
     * this function. This does not include dynamically-allocated shared
     * memory requested by the user at runtime.
     */
    SD_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1,

    /**
     * The size in bytes of user-allocated constant memory required by this
     * function.
     */
    SD_FUNC_ATTRIBUTE_CONST_SIZE_BYTES = 2,

    /**
     * The size in bytes of local memory used by each thread of this function.
     */
    SD_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3,

    /**
     * The number of registers used by each thread of this function.
     */
    SD_FUNC_ATTRIBUTE_NUM_REGS = 4,

    /**
     * The PTX virtual architecture version for which the function was
     * compiled. This value is the major PTX version * 10 + the minor PTX
     * version, so a PTX version 1.3 function would return the value 13.
     * Note that this may return the undefined value of 0 for cubins
     * compiled prior to SDAA 3.0.
     */
    SD_FUNC_ATTRIBUTE_PTX_VERSION = 5,

    /**
     * The binary architecture version for which the function was compiled.
     * This value is the major binary version * 10 + the minor binary version,
     * so a binary version 1.3 function would return the value 13. Note that
     * this will return a value of 10 for legacy cubins that do not have a
     * properly-encoded binary architecture version.
     */
    SD_FUNC_ATTRIBUTE_BINARY_VERSION = 6,

    SD_FUNC_ATTRIBUTE_MAX
} SDfunction_attribute;

/**
 * Memory types
 */
typedef enum SDmemorytype_enum {
    SD_MEMORYTYPE_HOST    = 0x01,    /**< Host memory */
    SD_MEMORYTYPE_DEVICE  = 0x02,    /**< Device memory */
    SD_MEMORYTYPE_UNIFIED = 0x04     /**< Unified device or host memory */
} SDmemorytype;

/**
 * Compute Modes
 */
typedef enum SDcomputemode_enum {
    SD_COMPUTEMODE_DEFAULT    = 0,  /**< Default compute mode (Multiple contexts allowed per device) */
    SD_COMPUTEMODE_EXCLUSIVE         = 1, /**< Compute-exclusive-thread mode (Only one context used by a single thread can be present on this device at a time) */
    SD_COMPUTEMODE_PROHIBITED        = 2, /**< Compute-prohibited mode (No contexts can be created on this device at this time) */
    SD_COMPUTEMODE_EXCLUSIVE_PROCESS = 3  /**< Compute-exclusive-process mode (Only one context used by a single process can be present on this device at a time) */
} SDcomputemode;

/**
 * Online compiler options
 */
typedef enum SDjit_option_enum
{
    /**
     * Max number of registers that a thread may use.\n
     * Option type: unsigned int
     */
    SD_JIT_MAX_REGISTERS = 0,

    /**
     * IN: Specifies minimum number of threads per block to target compilation
     * for\n
     * OUT: Returns the number of threads the compiler actually targeted.
     * This restricts the resource utilization fo the compiler (e.g. max
     * registers) such that a block with the given number of threads should be
     * able to launch based on register limitations. Note, this option does not
     * currently take into account any other resource limitations, such as
     * shared memory utilization.\n
     * Option type: unsigned int
     */
    SD_JIT_THREADS_PER_BLOCK,

    /**
     * Returns a float value in the option of the wall clock time, in
     * milliseconds, spent creating the cubin\n
     * Option type: float
     */
    SD_JIT_WALL_TIME,

    /**
     * Pointer to a buffer in which to print any log messsages from PTXAS
     * that are informational in nature (the buffer size is specified via
     * option ::SD_JIT_INFO_LOG_BUFFER_SIZE_BYTES) \n
     * Option type: char*
     */
    SD_JIT_INFO_LOG_BUFFER,

    /**
     * IN: Log buffer size in bytes.  Log messages will be capped at this size
     * (including null terminator)\n
     * OUT: Amount of log buffer filled with messages\n
     * Option type: unsigned int
     */
    SD_JIT_INFO_LOG_BUFFER_SIZE_BYTES,

    /**
     * Pointer to a buffer in which to print any log messages from PTXAS that
     * reflect errors (the buffer size is specified via option
     * ::SD_JIT_ERROR_LOG_BUFFER_SIZE_BYTES)\n
     * Option type: char*
     */
    SD_JIT_ERROR_LOG_BUFFER,

    /**
     * IN: Log buffer size in bytes.  Log messages will be capped at this size
     * (including null terminator)\n
     * OUT: Amount of log buffer filled with messages\n
     * Option type: unsigned int
     */
    SD_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,

    /**
     * Level of optimizations to apply to generated code (0 - 4), with 4
     * being the default and highest level of optimizations.\n
     * Option type: unsigned int
     */
    SD_JIT_OPTIMIZATION_LEVEL,

    /**
     * No option value required. Determines the target based on the current
     * attached context (default)\n
     * Option type: No option value needed
     */
    SD_JIT_TARGET_FROM_CUCONTEXT,

    /**
     * Target is chosen based on supplied ::SDjit_target_enum.\n
     * Option type: unsigned int for enumerated type ::SDjit_target_enum
     */
    SD_JIT_TARGET,

    /**
     * Specifies choice of fallback strategy if matching cubin is not found.
     * Choice is based on supplied ::SDjit_fallback_enum.\n
     * Option type: unsigned int for enumerated type ::SDjit_fallback_enum
     */
    SD_JIT_FALLBACK_STRATEGY

} SDjit_option;

/**
 * Online compilation targets
 */
typedef enum SDjit_target_enum
{
    SD_TARGET_COMPUTE_10 = 0,   /**< Compute device class 1.0 */
    SD_TARGET_COMPUTE_11,       /**< Compute device class 1.1 */
    SD_TARGET_COMPUTE_12,       /**< Compute device class 1.2 */
    SD_TARGET_COMPUTE_13,       /**< Compute device class 1.3 */
    SD_TARGET_COMPUTE_20,       /**< Compute device class 2.0 */
    SD_TARGET_COMPUTE_21,       /**< Compute device class 2.1 */
    SD_TARGET_COMPUTE_30        /**< Compute device class 3.0 */
} SDjit_target;

/**
 * Cubin matching fallback strategies
 */
typedef enum SDjit_fallback_enum
{
    SD_PREFER_PTX = 0,  /**< Prefer to compile ptx */
    SD_PREFER_BINARY    /**< Prefer to fall back to compatible binary code */

} SDjit_fallback;


/**
 * Limits
 */
typedef enum SDlimit_enum {
    SD_LIMIT_STACK_SIZE        = 0x00, /**< GPU thread stack size */
    SD_LIMIT_PRINTF_FIFO_SIZE  = 0x01, /**< GPU printf FIFO size */
    SD_LIMIT_MALLOC_HEAP_SIZE  = 0x02  /**< GPU malloc heap size */
} SDlimit;

/**
 * Interprocess Handles
 */
typedef enum SDipcMem_flags_enum {
    SD_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = 0x1  /** < Automatically enable peer access between remote devices as needed */
} SDipcMem_flags;

/**
 * Error codes
 */
typedef enum sdaaError_enum {
    /**
     * The API call returned with no errors. In the case of query calls, this
     * can also mean that the operation being queried is complete (see
     * ::sdEventQuery() and ::sdStreamQuery()).
     */
    SDAA_SUCCESS                              = 0,

    /**
     * This indicates that one or more of the parameters passed to the API call
     * is not within an acceptable range of values.
     */
    SDAA_ERROR_INVALID_VALUE                  = 1,

    /**
     * The API call failed because it was unable to allocate enough memory to
     * perform the requested operation.
     */
    SDAA_ERROR_OUT_OF_MEMORY                  = 2,

    /**
     * This indicates that the SDAA driver has not been initialized with
     * ::sdInit() or that initialization has failed.
     */
    SDAA_ERROR_NOT_INITIALIZED                = 3,

    /**
     * This indicates that the SDAA driver is in the process of shutting down.
     */
    SDAA_ERROR_DEINITIALIZED                  = 4,

    /**
     * This indicates profiling APIs are called while application is running
     * in visual profiler mode. 
    */
    SDAA_ERROR_PROFILER_DISABLED           = 5,
    /**
     * This indicates profiling has not been initialized for this context. 
     * Call cuProfilerInitialize() to resolve this. 
    */
    SDAA_ERROR_PROFILER_NOT_INITIALIZED       = 6,
    /**
     * This indicates profiler has already been started and probably
     * cuProfilerStart() is incorrectly called.
    */
    SDAA_ERROR_PROFILER_ALREADY_STARTED       = 7,
    /**
     * This indicates profiler has already been stopped and probably
     * cuProfilerStop() is incorrectly called.
    */
    SDAA_ERROR_PROFILER_ALREADY_STOPPED       = 8,  

    /**
     *  This indicates that no valid stream
     **/
    SDAA_ERROR_NO_STREAM                      = 9,
    /**
     *  This indicates that do not support 
     **/
    SDAA_ERROR_NO_SUPPORT                      = 10,
    /**
     * This indicates that no SDAA-capable devices were detected by the installed
     * SDAA driver.
     */
    SDAA_ERROR_NO_DEVICE                      = 100,

    /**
     * This indicates that the device ordinal supplied by the user does not
     * correspond to a valid SDAA device.
     */
    SDAA_ERROR_INVALID_DEVICE                 = 101,


    /**
     * This indicates that the device kernel image is invalid. This can also
     * indicate an invalid SDAA module.
     */
    SDAA_ERROR_INVALID_IMAGE                  = 200,

    /**
     * This most frequently indicates that there is no context bound to the
     * current thread. This can also be returned if the context passed to an
     * API call is not a valid handle (such as a context that has had
     * ::sdCtxDestroy() invoked on it). This can also be returned if a user
     * mixes different API versions (i.e. 3010 context with 3020 API calls).
     * See ::sdCtxGetApiVersion() for more details.
     */
    SDAA_ERROR_INVALID_CONTEXT                = 201,

    /**
     * This indicated that the context being supplied as a parameter to the
     * API call was already the active context.
     * \deprecated
     * This error return is deprecated as of SDAA 3.2. It is no longer an
     * error to attempt to push the active context via ::sdCtxPushCurrent().
     */
    SDAA_ERROR_CONTEXT_ALREADY_CURRENT        = 202,

    /**
     * This indicates that a map or register operation has failed.
     */
    SDAA_ERROR_MAP_FAILED                     = 205,

    /**
     * This indicates that an unmap or unregister operation has failed.
     */
    SDAA_ERROR_UNMAP_FAILED                   = 206,

    /**
     * This indicates that the specified array is currently mapped and thus
     * cannot be destroyed.
     */
    SDAA_ERROR_ARRAY_IS_MAPPED                = 207,

    /**
     * This indicates that the resource is already mapped.
     */
    SDAA_ERROR_ALREADY_MAPPED                 = 208,

    /**
     * This indicates that there is no kernel image available that is suitable
     * for the device. This can occur when a user specifies code generation
     * options for a particular SDAA source file that do not include the
     * corresponding device configuration.
     */
    SDAA_ERROR_NO_BINARY_FOR_GPU              = 209,

    /**
     * This indicates that a resource has already been acquired.
     */
    SDAA_ERROR_ALREADY_ACQUIRED               = 210,

    /**
     * This indicates that a resource is not mapped.
     */
    SDAA_ERROR_NOT_MAPPED                     = 211,

    /**
     * This indicates that a mapped resource is not available for access as an
     * array.
     */
    SDAA_ERROR_NOT_MAPPED_AS_ARRAY            = 212,

    /**
     * This indicates that a mapped resource is not available for access as a
     * pointer.
     */
    SDAA_ERROR_NOT_MAPPED_AS_POINTER          = 213,

    /**
     * This indicates that an uncorrectable ECC error was detected during
     * execution.
     */
    SDAA_ERROR_ECC_UNCORRECTABLE              = 214,

    /**
     * This indicates that the ::SDlimit passed to the API call is not
     * supported by the active device.
     */
    SDAA_ERROR_UNSUPPORTED_LIMIT              = 215,

    /**
     * This indicates that the ::SDcontext passed to the API call can
     * only be bound to a single CPU thread at a time but is already 
     * bound to a CPU thread.
     */
    SDAA_ERROR_CONTEXT_ALREADY_IN_USE         = 216,

    /**
     * This indicates that the device kernel source is invalid.
     */
    SDAA_ERROR_INVALID_SOURCE                 = 300,

    /**
     * This indicates that the file specified was not found.
     */
    SDAA_ERROR_FILE_NOT_FOUND                 = 301,

    /**
     * This indicates that a link to a shared object failed to resolve.
     */
    SDAA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302,

    /**
     * This indicates that initialization of a shared object failed.
     */
    SDAA_ERROR_SHARED_OBJECT_INIT_FAILED      = 303,

    /**
     * This indicates that an OS call failed.
     */
    SDAA_ERROR_OPERATING_SYSTEM               = 304,


    /**
     * This indicates that a resource handle passed to the API call was not
     * valid. Resource handles are opaque types like ::SDstream and ::SDevent.
     */
    SDAA_ERROR_INVALID_HANDLE                 = 400,


    /**
     * This indicates that a named symbol was not found. Examples of symbols
     * are global/constant variable names, texture names, and surface names.
     */
    SDAA_ERROR_NOT_FOUND                      = 500,


    /**
     * This indicates that asynchronous operations issued previously have not
     * completed yet. This result is not actually an error, but must be indicated
     * differently than ::SDAA_SUCCESS (which indicates completion). Calls that
     * may return this value include ::sdEventQuery() and ::sdStreamQuery().
     */
    SDAA_ERROR_NOT_READY                      = 600,


    /**
     * An exception occurred on the device while executing a kernel. Common
     * causes include dereferencing an invalid device pointer and accessing
     * out of bounds shared memory. The context cannot be used, so it must
     * be destroyed (and a new one should be created). All existing device
     * memory allocations from this context are invalid and must be
     * reconstructed if the program is to continue using SDAA.
     */
    SDAA_ERROR_LAUNCH_FAILED                  = 700,

    /**
     * This indicates that a launch did not occur because it did not have
     * appropriate resources. This error usually indicates that the user has
     * attempted to pass too many arguments to the device kernel, or the
     * kernel launch specifies too many threads for the kernel's register
     * count. Passing arguments of the wrong size (i.e. a 64-bit pointer
     * when a 32-bit int is expected) is equivalent to passing too many
     * arguments and can also result in this error.
     */
    SDAA_ERROR_LAUNCH_OUT_OF_RESOURCES        = 701,

    /**
     * This indicates that the device kernel took too long to execute. This can
     * only occur if timeouts are enabled - see the device attribute
     * ::SD_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT for more information. The
     * context cannot be used (and must be destroyed similar to
     * ::SDAA_ERROR_LAUNCH_FAILED). All existing device memory allocations from
     * this context are invalid and must be reconstructed if the program is to
     * continue using SDAA.
     */
    SDAA_ERROR_LAUNCH_TIMEOUT                 = 702,

    /**
     * This error indicates a kernel launch that uses an incompatible texturing
     * mode.
     */
    SDAA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING  = 703,
    
    /**
     * This error indicates that a call to ::cuCtxEnablePeerAccess() is
     * trying to re-enable peer access to a context which has already
     * had peer access to it enabled.
     */
    SDAA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 704,

    /**
     * This error indicates that ::cuCtxDisablePeerAccess() is 
     * trying to disable peer access which has not been enabled yet 
     * via ::cuCtxEnablePeerAccess(). 
     */
    SDAA_ERROR_PEER_ACCESS_NOT_ENABLED    = 705,

    /**
     * This error indicates that the primary context for the specified device
     * has already been initialized.
     */
    SDAA_ERROR_PRIMARY_CONTEXT_ACTIVE         = 708,

    /**
     * This error indicates that the context current to the calling thread
     * has been destroyed using ::sdCtxDestroy, or is a primary context which
     * has not yet been initialized.
     */
    SDAA_ERROR_CONTEXT_IS_DESTROYED           = 709,

    /**
     * A device-side assert triggered during kernel execution.
     * The context cannot be used anymore, and must be destroyed.
     * All existing device memory allocations from this context are invalid
     * and must be reconstructed if the program is to continue using SDAA.
     */
    SDAA_ERROR_ASSERT				= 710,

    /**
     * This error indicates that the hardware resources required to enable
     * peer access have been exhausted for one or more of the devices passed
     * to cuCtxEnablePeerAccess().
     */
    SDAA_ERROR_TOO_MANY_PEERS			= 711,

    /**
     * This error indicates that the memory range passed to sdMemHostRegister()
     * has already been registered.
     */
    SDAA_ERROR_HOST_MEMORY_ALREADY_REGISTERED	= 712,

    /**
     * This error indicates that the pointer passed to sdMemHostUnregister()
     * does not correspond to any currently registered memory region.
     */
    SDAA_ERROR_HOST_MEMORY_NOT_REGISTERED	= 713, 

    /**
     * This indicates that an unknown internal error has occurred.
     */
    SDAA_ERROR_UNKNOWN                        = 999
} SDresult;

/**
 * Interprocess Handles
 */
#define SD_IPC_HANDLE_SIZE		64

/**
 * End of array terminator for the extra parameter to sdLaunchKernel
 */
#define SD_LAUNCH_PARAM_END		((void*)0x00)

/**
 * Indicator that the next value in the extra parameter to sdLaunchKernel
 * will be a pointer to a buffer containing all kernel parameters used for
 * launching kernel f. This buffer needs to honor all alignment/padding
 * requirements of the individual parameters.
 * If SD_LAUNCH_PARAM_BUFFER_SIZE is not also specified in the extra array,
 * then SD_LAUNCH_PARAM_BUFFER_POINTER will have no effect. 
 */
#define SD_LAUNCH_PARAM_BUFFER_POINTER	((void*)0x01)

/**
 * Indicator that the next value in the extra parameter to sdLaunchKernel
 * will be a pointer to a size_t which contains the size of the buffer
 * specified with SD_LAUNCH_PARAM_BUFFER_POINTER.
 * It is required that SD_LAUNCH_PARAM_BUFFER_POINTER also be specified in
 * the extra array if the value associated with SD_LAUNCH_PARAM_BUFFER_SIZE
 * is not zero.
 */
#define SD_LAUNCH_PARAM_BUFFER_SIZE	((void*)0x02)

/**
 * If set, host memory is portable between SDAA contexts.
 * Flag for ::sdMemHostAlloc()
 */
#define SD_MEMHOSTALLOC_PORTABLE        0x01

/**
 * If set, host memory is mapped into SDAA address space and
 * ::sdMemHostGetDevicePointer() may be called on the host pointer.
 * Flag for ::sdMemHostAlloc()
 */
#define SD_MEMHOSTALLOC_DEVICEMAP       0x02

/**
 * If set, host memory is allocated as write-combined - fast to write,
 * faster to DMA, slow to read except via SSE4 streaming load instruction
 * (MOVNTDQA).
 * Flag for ::sdMemHostAlloc()
 */
#define SD_MEMHOSTALLOC_WRITECOMBINED   0x04

/**
 * If set, host memory is portable between SDAA contexts.
 * Flag for sdMemHostRegister()
 */
#define SD_MEMHOSTREGISTER_PORTABLE	0x01

/**
 * If set, host memory is mapped into SDAA address space and
 * sdMemHostGetDevicePointer() may be called on the host pointer.
 * Flag for sdMemHostRegister()
 */
#define SD_MEMHOSTREGISTER_DEVICEMAP	0x02

/**
 * For texture references loaded into the module, use default texunit from
 * texture reference.
 */
#define SD_PARAM_TR_DEFAULT		-1



/**
 * SDAA API version number
 */
#define SDAA_VERSION			4020

/* Initialization */
SDresult sdInit(unsigned int Flags);

/* Device Management */
SDresult sdDeviceComputeCapability(int *major, int *minor, SDdevice dev);
SDresult sdDeviceGet(SDdevice *device, int ordinal);
SDresult sdDeviceGetAttribute(int *pi, SDdevice_attribute attrib, SDdevice dev);
SDresult sdDeviceGetCount(int *count);
SDresult sdDeviceGetName(char *name, int len, SDdevice dev);
SDresult sdDeviceGetProperties(SDdevprop *prop, SDdevice dev);
SDresult sdDeviceTotalMem(size_t *bytes, SDdevice dev);
SDresult sdDeviceGetState(int *device_state, int *fault_type, unsigned long *cg_fault_reg);

/* Event Management */
SDresult sdEventCreate(SDevent *phEvent, unsigned int Flags);
SDresult sdEventDestroy(SDevent hEvent);
SDresult sdEventElapsedTime(float *pMilliseconds, SDevent hStart, SDevent hEnd);
SDresult sdEventQuery(SDevent hEvent);
SDresult sdEventRecord(SDevent hEvent, SDstream hStream);
SDresult sdEventSynchronize(SDevent hEvent);

/* Version Management */
SDresult sdDriverGetVersion (int *driverVersion);

/* Context Management */
SDresult sdCtxAttach(SDcontext *pctx, unsigned int flags);
SDresult sdCtxCreate(SDcontext *pctx, unsigned int flags, SDdevice dev);
SDresult sdCtxDestroy(SDcontext ctx);
SDresult sdCtxDetach(SDcontext ctx);
SDresult sdCtxGetApiVersion(SDcontext ctx, unsigned int *version);
SDresult sdCtxGetCurrent(SDcontext *pctx);
SDresult sdCtxGetDevice(SDdevice *device);
SDresult sdCtxGetLimit(size_t *pvalue, SDlimit limit);
SDresult sdCtxPopCurrent(SDcontext *pctx);
SDresult sdCtxPushCurrent(SDcontext ctx);
SDresult sdCtxSetCurrent(SDcontext ctx);
SDresult sdCtxSetLimit(SDlimit limit, size_t value);
SDresult sdCtxSynchronize(void);

/* Module Management */
SDresult sdModuleGetFunction(SDfunction *hfunc, SDmodule hmod, const char *name);
SDresult sdModuleLoad(SDmodule *module, const char *fname);
SDresult sdModuleLoadData(SDmodule *module, const void *image);
SDresult sdModuleLoadDataEx(SDmodule *module, const void *image, unsigned int numOptions, SDjit_option *options, void **optionValues);
SDresult sdModuleLoadFatBinary(SDmodule *module, const void *fatCubin);
SDresult sdModuleUnload(SDmodule hmod);

/* Execution Control */
SDresult sdFuncGetAttribute(int *pi, SDfunction_attribute attrib, SDfunction hfunc);
SDresult sdFuncSetBlockShape(SDfunction hfunc, int x, int y, int z);
SDresult sdFuncSetSharedSize(SDfunction hfunc, unsigned int bytes);
SDresult sdLaunch(SDfunction f, int num, SDstream hStream);
SDresult sdPrint(void* hostaddr, unsigned long size, unsigned int type, SDstream hStream);
SDresult sdLaunchGrid(SDfunction f, int grid_width, int grid_height);
SDresult sdLaunchGridAsync(SDfunction f, int grid_width, int grid_height, SDstream hStream);
SDresult sdLaunchKernel(SDfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, SDstream hStream, void **kernelParams, void **extra);
SDresult sdParamSetf(SDfunction hfunc, int offset, double value);
SDresult sdParamSeti(SDfunction hfunc, int offset, size_t value);
SDresult sdParamSetSize(SDfunction hfunc, unsigned int numbytes);
SDresult sdParamSetv(SDfunction hfunc, int offset, void *ptr, unsigned int numbytes);

/* Memory Management (Incomplete) */
SDresult sdMemAlloc(SDdeviceptr *dptr, unsigned long bytesize);
SDresult sdMemFree(SDdeviceptr dptr);
SDresult sdMemAllocHost(void **pp, unsigned int bytesize);
SDresult sdMemFreeHost(void *p);
SDresult sdMemcpyDtoH(void *dstHost, SDdeviceptr srcDevice, unsigned int ByteCount);
SDresult sdMemcpyDtoHAsync(void *dstHost, SDdeviceptr srcDevice, unsigned int ByteCount, SDstream hStream);
SDresult sdMemcpyHtoD(SDdeviceptr dstDevice, const void *srcHost, unsigned int ByteCount);
SDresult sdMemcpyHtoDAsync(SDdeviceptr dstDevice, const void *srcHost, unsigned int ByteCount, SDstream hStream);
SDresult sdMemcpyDtoD(SDdeviceptr dstDevice, SDdeviceptr srcDevice, unsigned int ByteCount);
SDresult sdMemHostAlloc(void **pp, unsigned int bytesize, unsigned int Flags);
SDresult sdMemHostGetDevicePointer(SDdeviceptr *pdptr, void *p, unsigned int Flags);

SDresult sdMemsetD8(SDdeviceptr dstDevice, unsigned char uc, size_t N);
SDresult sdMemsetD16(SDdeviceptr dstDevice, unsigned short us, size_t N);
SDresult sdMemsetD32(SDdeviceptr dstDevice, unsigned int ui, size_t N);
#if 0
/* Memory mapping - Gdev extension */
SDresult sdMemMap(void **buf, SDdeviceptr dptr, unsigned int bytesize);
SDresult sdMemUnmap(void *buf);
#endif
/* Memory mapped address - Gdev extension */
SDresult sdMemGetPhysAddr(unsigned long long *addr, void *p);

/* Stream Management */
SDresult sdStreamCreate(SDstream *phStream, unsigned int Flags);
SDresult sdStreamDestroy(SDstream hStream);
SDresult sdStreamQuery(SDstream hStream);
SDresult sdStreamSynchronize(SDstream hStream);
SDresult sdStreamWaitEvent(SDstream hStream, SDevent hEvent, unsigned int Flags);

#if 0
/* Inter-Process Communication (IPC) - Gdev extension */
SDresult sdShmGet(int *ptr, int key, size_t size, int flags);
SDresult sdShmAt(SDdeviceptr *dptr, int id, int flags);
SDresult sdShmDt(SDdeviceptr dptr);
SDresult sdShmCtl(int id, int cmd, void *buf /* FIXME */);
#endif

/*IPC memory handle*/

SDresult sdMemGetP2PAddr(void **p2p_addr, void *p);

SDresult sdIpcGetMemHandle(int *mem_handle, void *devPtr);
SDresult sdIpcOpenMemHandle (void** devPtr, int mem_handle, int flags);
SDresult sdIpcCloseMemHandle (void* devPtr);
size_t sdIpcGetMemSize (void* devPtr);

#ifdef __cplusplus
}
#endif /* __cplusplus  */
#endif
