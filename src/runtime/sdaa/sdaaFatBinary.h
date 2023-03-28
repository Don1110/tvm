// /*!
// 	\file sdaaFatBinary.h
// 	\author Andrew Kerr <arkerr@gatech.edu>
// 	\brief this was extracted from sdaa_runtime.h as these structures are shared by
// 		both the SDAA Runtime API and Driver APIs
// */

#ifndef OCELOT_SDAA_SDAAFATBINARY_H_INCLUDED
#define OCELOT_SDAA_SDAAFATBINARY_H_INCLUDED

/*----------------------------------- Types ----------------------------------*/

/*
 * Cubin entry type for __sdaaFat binary. 
 * Cubins are specific to a particular gpu profile,
 * although the gpuInfo module might 'know'
 * that cubins will also run on other gpus.
 * Based on the recompilation strategy, 
 * fatGetCubinForGpu will return an existing
 * compatible load image, or attempt a recompilation.
 */
typedef struct {
    char*            gpuProfileName;
    char*            cubin;
} __sdaaFatCubinEntry;


/*
 * Ptx entry type for __sdaaFat binary.
 * PTX might use particular chip features
 * (such as double precision floating points).
 * When attempting to recompile for a certain 
 * gpu architecture, a ptx needs to be available
 * that depends on features that are either 
 * implemented by the gpu, or for which the ptx
 * translator can provide an emulation. 
 */
typedef struct {
    char*            gpuProfileName;            
    char*            ptx;
} __sdaaFatPtxEntry;


/*
 * Debug entry type for __sdaaFat binary.
 * Such information might, but need not be available
 * for Cubin entries (ptx files compiled in debug mode
 * will contain their own debugging information) 
 */
typedef struct __sdaaFatDebugEntryRec {
    char*                   gpuProfileName;            
    char*                   debug;
    struct __sdaaFatDebugEntryRec *next;
    unsigned int                   size;
} __sdaaFatDebugEntry;

typedef struct __sdaaFatElfEntryRec {
    char*                 gpuProfileName;            
    char*                 elf;
    struct __sdaaFatElfEntryRec *next;
    unsigned int                 size;
} __sdaaFatElfEntry;

typedef enum {
      __sdaaFatDontSearchFlag = (1 << 0),
      __sdaaFatDontCacheFlag  = (1 << 1),
      __sdaaFatSassDebugFlag  = (1 << 2)
} __sdaaFatSdaaBinaryFlag;

/*
 * Imported/exported symbol descriptor, needed for 
 * __sdaaFat binary linking. Not much information is needed,
 * because this is only an index: full symbol information 
 * is contained by the binaries.
 */
typedef struct {
    char* name;
} __sdaaFatSymbol;

/*
 * Fat binary container.
 * A mix of ptx intermediate programs and cubins,
 * plus a global identifier that can be used for 
 * further lookup in a translation cache or a resource
 * file. This key is a checksum over the device text.
 * The ptx and cubin array are each terminated with 
 * entries that have NULL components.
 */
 
typedef struct __sdaaFatSdaaBinaryRec {
    unsigned long            magic;
    unsigned long            version;
    unsigned long            gpuInfoVersion;
    char*                   key;
    char*                   ident;
    char*                   usageMode;
    __sdaaFatPtxEntry             *ptx;
    __sdaaFatCubinEntry           *cubin;
    __sdaaFatDebugEntry           *debug;
    void*                  debugInfo;
    unsigned int                   flags;
    __sdaaFatSymbol               *exported;
    __sdaaFatSymbol               *imported;
    struct __sdaaFatSdaaBinaryRec *dependends;
    unsigned int                   characteristic;
    __sdaaFatElfEntry             *elf;
} __sdaaFatSdaaBinary;

typedef struct __sdaaFatSdaaBinary2HeaderRec { 
    unsigned int            magic;
    unsigned int            version;
	unsigned long long int  length;
} __sdaaFatSdaaBinary2Header;

enum FatBin2EntryType {
	FATBIN_2_PTX = 0x1,
	FATBIN_2_ELF = 0x2,
	FATBIN_2_OLDCUBIN = 0x4
};

typedef struct __sdaaFatSdaaBinary2EntryRec { 
	unsigned int           type;
	unsigned int           binary;
	unsigned long long int binarySize;
	unsigned int           unknown2;
	unsigned int           kindOffset;
	unsigned int           unknown3;
	unsigned int           unknown4;
	unsigned int           name;
	unsigned int           nameSize;
	unsigned long long int flags;
	unsigned long long int unknown7;
	unsigned long long int uncompressedBinarySize;
} __sdaaFatSdaaBinary2Entry;

#define COMPRESSED_PTX 0x0000000000001000LL

typedef struct __sdaaFatSdaaBinaryRec2 {
	int magic;
	int version;
	const unsigned long long* fatbinData;
	char* f;
} __sdaaFatSdaaBinary2;

/*
 * Current version and magic numbers:
 */
#define __sdaaFatVERSION   0x00000004
#define __sdaaFatMAGIC     0x1ee55a01
#define __sdaaFatMAGIC2    0x466243b1
#define __sdaaFatMAGIC3    0xba55ed50

/*
 * Version history log:
 *    1  : __sdaaFatDebugEntry field added to __sdaaFatSdaaBinary struct
 *    2  : flags and debugInfo field added.
 *    3  : import/export symbol list
 *    4  : characteristic added, elf added
 */


/*--------------------------------- Functions --------------------------------*/

typedef enum {
    __sdaaFatAvoidPTX,
    __sdaaFatPreferBestCode,
    __sdaaFatForcePTX
} __sdaaFatCompilationPolicy;

#endif

