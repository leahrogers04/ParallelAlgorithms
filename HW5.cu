// Name:
// nvcc HW5.cu -o temp
/*
 What to do:
 This code prints out useful information about the GPU(s) in your machine, 
 but there is much more data available in the cudaDeviceProp structure.

 Extend this code so that it prints out all the information about the GPU(s) in your system. 
 Also, and this is the fun part, be prepared to explain what each piece of information means. 
*/

// Include files
#include <stdio.h>

// Defines

// Global variables

// Function prototypes
void cudaErrorCheck(const char*, int);

void cudaErrorCheck(const char *file, int line)
{
	cudaError_t  error;
	error = cudaGetLastError();

	if(error != cudaSuccess)
	{
		printf("\n CUDA ERROR: message = %s, File = %s, Line = %d\n", cudaGetErrorString(error), file, line);
		exit(0);
	}
}

int main()
{
	cudaDeviceProp prop;

	int count;
	cudaGetDeviceCount(&count);
	cudaErrorCheck(__FILE__, __LINE__);
	printf(" You have %d GPUs in this machine\n", count);
	
	for (int i=0; i < count; i++) {
		cudaGetDeviceProperties(&prop, i);
		cudaErrorCheck(__FILE__, __LINE__);
		printf(" ---General Information for device %d ---\n", i);
		printf("Name: %s\n", prop.name);
		printf("Compute capability: %d.%d\n", prop.major, prop.minor);
		printf("Clock rate: %d\n", prop.clockRate);
		printf("Device copy overlap: ");
		if (prop.deviceOverlap) printf("Enabled\n");
		else printf("Disabled\n");
		printf("Kernel execution timeout : ");
		if (prop.kernelExecTimeoutEnabled) printf("Enabled\n");
		else printf("Disabled\n");
		printf(" ---Memory Information for device %d ---\n", i);
		printf("Total global mem: %ld\n", prop.totalGlobalMem);
		printf("Total constant Mem: %ld\n", prop.totalConstMem);
		printf("Max mem pitch: %ld\n", prop.memPitch);
		printf("Texture Alignment: %ld\n", prop.textureAlignment);
		printf(" ---MP Information for device %d ---\n", i);
		printf("Multiprocessor count : %d\n", prop.multiProcessorCount); //Number of SMs on the GPU
		printf("Shared mem per mp: %ld\n", prop.sharedMemPerBlock);
		printf("Registers per mp: %d\n", prop.regsPerBlock);
		printf("Threads in warp: %d\n", prop.warpSize);
		printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
		printf("Max thread dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
		printf("Max grid dimensions: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
		printf("\n");
		// additional information I added
		 printf(" ---Additional Information for device %d ---\n", i);
        printf("Access policy max window size: %d\n", prop.accessPolicyMaxWindowSize);
        printf("Can use host pointer for registered memory: %s\n", prop.canUseHostPointerForRegisteredMem ? "Yes" : "No");
        printf("Compute preemption supported: %s\n", prop.computePreemptionSupported ? "Yes" : "No");
        printf("Concurrent managed access: %s\n", prop.concurrentManagedAccess ? "Yes" : "No");
        printf("Cooperative launch: %s\n", prop.cooperativeLaunch ? "Yes" : "No");
        printf("Cooperative multi-device launch: %s\n", prop.cooperativeMultiDeviceLaunch ? "Yes" : "No");
        printf("Host native atomic supported: %s\n", prop.hostNativeAtomicSupported ? "Yes" : "No");
        printf("LUID: %s\n", prop.luid);
        printf("LUID device node mask: %u\n", prop.luidDeviceNodeMask);
        printf("Max blocks per multiprocessor: %d\n", prop.maxBlocksPerMultiProcessor);
        printf("Max surface 1D: %d\n", prop.maxSurface1D);
        printf("Max surface 1D layered: (%d, %d)\n", prop.maxSurface1DLayered[0], prop.maxSurface1DLayered[1]);
        printf("Max surface 2D: (%d, %d)\n", prop.maxSurface2D[0], prop.maxSurface2D[1]);
        printf("Max surface 2D layered: (%d, %d, %d)\n", prop.maxSurface2DLayered[0], prop.maxSurface2DLayered[1], prop.maxSurface2DLayered[2]);
        printf("Max surface 3D: (%d, %d, %d)\n", prop.maxSurface3D[0], prop.maxSurface3D[1], prop.maxSurface3D[2]);
        printf("Max surface cubemap: %d\n", prop.maxSurfaceCubemap);
        printf("Max surface cubemap layered: (%d, %d)\n", prop.maxSurfaceCubemapLayered[0], prop.maxSurfaceCubemapLayered[1]);
        printf("Max texture 1D: %d\n", prop.maxTexture1D);
        printf("Max texture 1D layered: (%d, %d)\n", prop.maxTexture1DLayered[0], prop.maxTexture1DLayered[1]);
        printf("Max texture 1D linear: %d\n", prop.maxTexture1DLinear);
        printf("Max texture 1D mipmap: %d\n", prop.maxTexture1DMipmap);
        printf("Max texture 2D: (%d, %d)\n", prop.maxTexture2D[0], prop.maxTexture2D[1]);
        printf("Max texture 2D gather: (%d, %d)\n", prop.maxTexture2DGather[0], prop.maxTexture2DGather[1]);
        printf("Max texture 2D layered: (%d, %d, %d)\n", prop.maxTexture2DLayered[0], prop.maxTexture2DLayered[1], prop.maxTexture2DLayered[2]);
        printf("Max texture 2D linear: (%d, %d, %d)\n", prop.maxTexture2DLinear[0], prop.maxTexture2DLinear[1], prop.maxTexture2DLinear[2]);
        printf("Max texture 2D mipmap: (%d, %d)\n", prop.maxTexture2DMipmap[0], prop.maxTexture2DMipmap[1]);
        printf("Max texture 3D: (%d, %d, %d)\n", prop.maxTexture3D[0], prop.maxTexture3D[1], prop.maxTexture3D[2]);
        printf("Max texture 3D alt: (%d, %d, %d)\n", prop.maxTexture3DAlt[0], prop.maxTexture3DAlt[1], prop.maxTexture3DAlt[2]);
        printf("Max texture cubemap: %d\n", prop.maxTextureCubemap);
        printf("Max texture cubemap layered: (%d, %d)\n", prop.maxTextureCubemapLayered[0], prop.maxTextureCubemapLayered[1]);
        printf("Pageable memory access: %s\n", prop.pageableMemoryAccess ? "Yes" : "No");
        printf("Pageable memory access uses host page tables: %s\n", prop.pageableMemoryAccessUsesHostPageTables ? "Yes" : "No");
        printf("Persisting L2 cache max size: %d\n", prop.persistingL2CacheMaxSize);
        printf("Reserved shared memory per block: %ld\n", prop.reservedSharedMemPerBlock);
        printf("Shared memory per block opt-in: %ld\n", prop.sharedMemPerBlockOptin);
        printf("Single to double precision performance ratio: %d\n", prop.singleToDoublePrecisionPerfRatio);
        printf("Surface alignment: %ld\n", prop.surfaceAlignment);
        printf("Texture pitch alignment: %ld\n", prop.texturePitchAlignment);
         printf("UUID: ");
        for (int j = 0; j < 16; j++) 
		{
            printf("%02x", (unsigned char)prop.uuid.bytes[j]);
        }
        printf("Warp size: %d\n", prop.warpSize);
        printf("\n");
		//printf("Cluster launch: %s\n", prop.clusterLaunch ? "Yes" : "No");
		//printf("Deferred mapping CUDA array supported: %s\n", prop.deferredMappingCudaArraySupported ? "Yes" : "No");
        //printf("Direct managed memory access from host: %s\n", prop.directManagedMemAccessFromHost ? "Yes" : "No");
        //printf("GPU direct RDMA flush writes options: %u\n", prop.gpuDirectRDMAFlushWritesOptions);
        //printf("GPU direct RDMA supported: %s\n", prop.gpuDirectRDMASupported ? "Yes" : "No");
        //printf("GPU direct RDMA writes ordering: %d\n", prop.gpuDirectRDMAWritesOrdering);
		//printf("Host register read-only supported: %s\n", prop.hostRegisterReadOnlySupported ? "Yes" : "No");
        //printf("Host register supported: %s\n", prop.hostRegisterSupported ? "Yes" : "No");
        //printf("IPC event supported: %s\n", prop.ipcEventSupported ? "Yes" : "No");
		//printf("Memory pool supported handle types: %u\n", prop.memoryPoolSupportedHandleTypes);
        //printf("Memory pools supported: %s\n", prop.memoryPoolsSupported ? "Yes" : "No");
		//printf("Reserved: %d\n", prop.reserved[0]); // Example for reserved array
		//printf("Sparse CUDA array supported: %s\n", prop.sparseCudaArraySupported ? "Yes" : "No");
		//printf("Timeline semaphore interop supported: %s\n", prop.timelineSemaphoreInteropSupported ? "Yes" : "No");
        //printf("Unified function pointers: %s\n", prop.unifiedFunctionPointers ? "Yes" : "No");
	}	
	return(0);
}

