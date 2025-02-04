// Name:Leah Rogers
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
		printf("Name: %s\n", prop.name); //Prints the name of the GPU.
		printf("Compute capability: %d.%d\n", prop.major, prop.minor);//Prints the compute capability of the GPU (major and minor versions).
		printf("Clock rate: %d\n", prop.clockRate);
		printf("Device copy overlap: "); // Checks if the device can perform memory copies and kernel execution concurrently.
		
		
		
		if (prop.deviceOverlap) printf("Enabled\n");//Enabled: The GPU supports concurrent copy and execution.
		else printf("Disabled\n");//Disabled: The GPU executes copy and kernel tasks sequentially.

		printf("Kernel execution timeout : "); //If enabled, CUDA kernels running too long (on GPUs connected to a display) will time out.
		//This prevents system freezes when using GPUs for both display and computation.
		if (prop.kernelExecTimeoutEnabled) printf("Enabled\n");
		else printf("Disabled\n");
		printf(" ---Memory Information for device %d ---\n", i);
		printf("Total global mem: %ld\n", prop.totalGlobalMem);//Prints total global memory available on the device in bytes

		printf("Total constant Mem: %ld\n", prop.totalConstMem);//The amount of constant memory, which is small but optimized for fast read access.
		printf("Max mem pitch: %ld\n", prop.memPitch); //the number of bytes you need to skip in memory to move to the next row of a 2D data structure
		printf("Texture Alignment: %ld\n", prop.textureAlignment);//The memory alignment requirement for textures in CUDA.
		//Textures should be aligned to this value for optimal performance.
		printf(" ---MP Information for device %d ---\n", i);

		printf("Multiprocessor count : %d\n", prop.multiProcessorCount); //Number of SMs on the GPU
		//The number of Streaming Multiprocessors (SMs) on the GPU.
		//More SMs = More parallel processing power.

		printf("Shared mem per mp: %ld\n", prop.sharedMemPerBlock); //memory shared between the threads in a block
		printf("Registers per mp: %d\n", prop.regsPerBlock); //memory associated with individual threads
		printf("Threads in warp: %d\n", prop.warpSize);//warp is a set of 32 threads within a thread block such that all the threads in a warp execute the same instruction
		printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
		printf("Max thread dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]); //max threads per block in each dimension
		printf("Max grid dimensions: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]); // max number of blocks in the grid for each dimension
		printf("\n");
		// additional information I added
		 printf(" ---Additional Information for device %d ---\n", i);
        printf("Access policy max window size: %d\n", prop.accessPolicyMaxWindowSize); //Maximum window size for access policy features (used in memory handling optimizations)

        printf("Can use host pointer for registered memory: %s\n", prop.canUseHostPointerForRegisteredMem ? "Yes" : "No");//ndicates if host pointers (allocated using cudaHostRegister) can be used in CUDA kernels.
	//If "Yes", it allows direct memory access without copying to GPU.
        printf("Compute preemption supported: %s\n", prop.computePreemptionSupported ? "Yes" : "No");//If "Yes", the GPU supports preemption, meaning kernels can be paused and resumed later.
	//Useful for multi-tasking and preventing long CUDA computations from blocking other GPU tasks.

        printf("Concurrent managed access: %s\n", prop.concurrentManagedAccess ? "Yes" : "No");//If "Yes", CUDA Unified Memory can be accessed concurrently from both CPU and GPU.
	//Enables efficient memory management between host and device.
        printf("Cooperative launch: %s\n", prop.cooperativeLaunch ? "Yes" : "No");//If "Yes", CUDA cooperative groups can be used.
	//Allows multiple thread blocks to communicate and synchronize across the entire GPU.

        printf("Cooperative multi-device launch: %s\n", prop.cooperativeMultiDeviceLaunch ? "Yes" : "No");//If "Yes", allows launching cooperative kernels across multiple GPUs.
	//Useful for multi-GPU computing where different GPUs collaborate on a task.

        printf("Host native atomic supported: %s\n", prop.hostNativeAtomicSupported ? "Yes" : "No");//If "Yes", the GPU supports atomic operations directly between CPU and GPU.
	//Helps with concurrent access to shared memory locations.
	//atomic operations are low-level programming operations that read, modify, and write data to memory without interference from other threads
        printf("LUID: %s\n", prop.luid); //Locally Unique Identifier (LUID) assigned to the GPU by the OS.
	//Used to distinguish between different GPUs in a system.

        printf("LUID device node mask: %u\n", prop.luidDeviceNodeMask);//Bitmask that helps identify which device nodes the GPU belongs to.
	//Relevant in multi-GPU environments.
	// bitmask is essentially an integer value used to represent a set of flags or conditions by manipulating individual bits within its binary representation, 
	//allowing for efficient storage and manipulation of multiple Boolean states within a single variable, especially when dealing with large datasets on the GPU

        printf("Max blocks per multiprocessor: %d\n", prop.maxBlocksPerMultiProcessor);//Maximum number of blocks that a single Streaming Multiprocessor (SM) can handle.
	//Affects occupancy and performance tuning.

        printf("Max surface 1D: %d\n", prop.maxSurface1D); //max width of 1d surface
        printf("Max surface 1D layered: (%d, %d)\n", prop.maxSurface1DLayered[0], prop.maxSurface1DLayered[1]);//max widtch and layer count for layered 1d surface
        printf("Max surface 2D: (%d, %d)\n", prop.maxSurface2D[0], prop.maxSurface2D[1]);//max width and heigh for 2d surface
        printf("Max surface 2D layered: (%d, %d, %d)\n", prop.maxSurface2DLayered[0], prop.maxSurface2DLayered[1], prop.maxSurface2DLayered[2]);//max width heigh and layer count for layered 2d surface
        printf("Max surface 3D: (%d, %d, %d)\n", prop.maxSurface3D[0], prop.maxSurface3D[1], prop.maxSurface3D[2]);//max width height and depth for 3d
        printf("Max surface cubemap: %d\n", prop.maxSurfaceCubemap); //max size for cubemap surface. the cubemap is a type of texture that has 6 2d cube faces used to capture the effect of a 3d enviroment
        printf("Max surface cubemap layered: (%d, %d)\n", prop.maxSurfaceCubemapLayered[0], prop.maxSurfaceCubemapLayered[1]);//max size and layer count for layered cubemap surfaces
        printf("Max texture 1D: %d\n", prop.maxTexture1D);//max width of 1d texture
        printf("Max texture 1D layered: (%d, %d)\n", prop.maxTexture1DLayered[0], prop.maxTexture1DLayered[1]);//Maximum width and layer count for 1D layered textures.
        printf("Max texture 1D linear: %d\n", prop.maxTexture1DLinear);
        printf("Max texture 1D mipmap: %d\n", prop.maxTexture1DMipmap);
        printf("Max texture 2D: (%d, %d)\n", prop.maxTexture2D[0], prop.maxTexture2D[1]);
        printf("Max texture 2D gather: (%d, %d)\n", prop.maxTexture2DGather[0], prop.maxTexture2DGather[1]);
        printf("Max texture 2D layered: (%d, %d, %d)\n", prop.maxTexture2DLayered[0], prop.maxTexture2DLayered[1], prop.maxTexture2DLayered[2]);
        printf("Max texture 2D linear: (%d, %d, %d)\n", prop.maxTexture2DLinear[0], prop.maxTexture2DLinear[1], prop.maxTexture2DLinear[2]);//Maximum width, height, and pitch for 2D linear textures.
        printf("Max texture 2D mipmap: (%d, %d)\n", prop.maxTexture2DMipmap[0], prop.maxTexture2DMipmap[1]);
        printf("Max texture 3D: (%d, %d, %d)\n", prop.maxTexture3D[0], prop.maxTexture3D[1], prop.maxTexture3D[2]);
        printf("Max texture 3D alt: (%d, %d, %d)\n", prop.maxTexture3DAlt[0], prop.maxTexture3DAlt[1], prop.maxTexture3DAlt[2]);
        printf("Max texture cubemap: %d\n", prop.maxTextureCubemap);
        printf("Max texture cubemap layered: (%d, %d)\n", prop.maxTextureCubemapLayered[0], prop.maxTextureCubemapLayered[1]);

        printf("Pageable memory access: %s\n", prop.pageableMemoryAccess ? "Yes" : "No");//Indicates whether the GPU supports direct access to pageable memory
	//(memory that can be swapped between RAM and disk by the OS)
        printf("Pageable memory access uses host page tables: %s\n", prop.pageableMemoryAccessUsesHostPageTables ? "Yes" : "No");//If "Yes", the GPU can leverage the host’s page tables instead of managing its own.
	//this Reduces memory duplication and overhead, Allows the GPU to efficiently track memory allocations in system RAM and Can enable faster memory access in Unified Memory (UVM) scenarios.
       
	printf("Persisting L2 cache max size: %d\n", prop.persistingL2CacheMaxSize);//The maximum amount of L2 cache memory that can persist across kernel launches.
	//The L2 cache is a fast on-GPU memory that holds frequently used data.
	//If the cache persists, it means data can survive across multiple kernel launches, reducing memory transfers from global memory and improving performance.

        printf("Reserved shared memory per block: %ld\n", prop.reservedSharedMemPerBlock);
        printf("Shared memory per block opt-in: %ld\n", prop.sharedMemPerBlockOptin);
        printf("Single to double precision performance ratio: %d\n", prop.singleToDoublePrecisionPerfRatio);//Ratio of single-precision (float) to double-precision (double) performance.
	//Some GPUs favor single-precision calculations, meaning they execute them faster than double-precision.

        printf("Surface alignment: %ld\n", prop.surfaceAlignment);//The alignment (in bytes) required for CUDA surface memory.

        printf("Texture pitch alignment: %ld\n", prop.texturePitchAlignment);//The required alignment for pitched textures.
         printf("UUID: ");//Used to identify specific GPUs in multi-GPU setups.
	//Ensures the correct GPU is selected for computations in multi-GPU clusters.
        for (int j = 0; j < 16; j++) 
		{
            printf("%02x", (unsigned char)prop.uuid.bytes[j]);
        }
        printf("Warp size: %d\n", prop.warpSize);// number of threads in a warp. all threads in a warp execute the same instruction
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

