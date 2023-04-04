
#ifndef INCLUDE_SDAA_TECO_DETAIL_HOST_DEFINES_H_
#define INCLUDE_SDAA_TECO_DETAIL_HOST_DEFINES_H_

// Add guard to Generic Grid Launch method
#ifndef GENERIC_GRID_LAUNCH
#define GENERIC_GRID_LAUNCH 1
#endif

#if defined(__clang__) && defined(__HIP__)

#if !__CLANG_HIP_RUNTIME_WRAPPER_INCLUDED__
#define __host__ __attribute__((host))
#define __device__ __attribute__((device))
#define __global__ __attribute__((global))
#define __shared__ __attribute__((shared))
#define __constant__ __attribute__((constant))
#endif  // !__CLANG_HIP_RUNTIME_WRAPPER_INCLUDED__

#define __noinline__ __attribute__((noinline))
#define __forceinline__ inline __attribute__((always_inline))

#else

/**
 * Function and kernel markers
 */
#define __host__
#define __device__

#define __global__

#define __noinline__
#define __forceinline__ inline

#define __shared__
#define __constant__

#endif

#endif //INCLUDE_SDAA_TECO_DETAIL_HOST_DEFINES_H_
