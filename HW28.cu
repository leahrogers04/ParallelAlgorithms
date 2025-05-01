// Name: Leah Rogers
// CPU random walk. 
// nvcc HW28.cu -o temp - lcurand

/*
 What to do:
 This is some code that runs a random walk for 10000 steps.
 Use cudaRand and run 10 of these runs at once with diferent seeds on the GPU.
 Print out all 10 final positions.
*/

// Include files
#include <sys/time.h>
#include <stdio.h>
#include <curand_kernel.h>

// Defines
#define WALKS 10
#define STEPS 10000
// Globals
float MidPoint = (float)UINT_MAX/2.0f; //using UINT_MAX bc curand() generates numbers from 0 to UINT_MAX so using RAND_MAX wouldnt correctly represmt the midpoint of the range 0 to UINT_MAX.
int *GPUfinalPositions; // Pointer for final positions on GPU
int *CPUfinalPositions;
dim3 BlockSize;
dim3 GridSize;

// Function prototypes
void cudaErrorCheck(const char *, int);
void setupDevices();
__global__ void randomWalk(int, unsigned long long, int, float);
int main(int, char**);

void cudaErrorCheck(const char *file, int line)
{
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

void setupDevices()
{
	BlockSize.x = WALKS; // Number of threads per block
	BlockSize.y = 1; 
	BlockSize.z = 1; 
	
	GridSize.x = 1; 
	GridSize.y = 1;
	GridSize.z = 1;

	//allocate memory on the CPU for an array that will store the final positions
	//of each random walk after the GPU has finished its computations

	//type casted to (int *) because the result of malloc is a void * and so you have to do that to indicate the allocated mem will be used as an array of integers.
	// malloc allocates memory on the CPU heap. 
	//WALKS is 10 so it alllocates mem for an array of 10 ints
	//sizeof(int) ensures the correct number of bytes are allocated for each int
	CPUfinalPositions = (int *)malloc(WALKS * sizeof(int)); 

	//cudaMalloc allocates memory on the GPU heap.
	//GPUfinalPositions is a pointer to an int that hold the adress of the allocated mem on the GPU.
	// & is used to pass the adress of the pointer to cudaMAlloc so it can update it w the adress of the allocated mem on the GPU.
	//WALKS * sizeof(int) is the number of bytes to allocate on the GPU.
	cudaMalloc(&GPUfinalPositions, WALKS * sizeof(int)); // Allocate memory on the GPU for final positions
	cudaErrorCheck(__FILE__, __LINE__); // Check for errors after memory allocation
}

__global__ void randomWalk(int *finalPositions, unsigned long long seed, int numberOfRandomSteps, float midPoint)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x; //blockIdx.x * blockDim.x calculates the offset of the blocks making sure the threads in different blocks have unique ids.
	// adding threadIdx.x gives the global thread id for each thread in the block.

	if (id < WALKS) // ensures only threads with ID less that WALKS execute the random walk logic
	{
		curandState state; //curandState is a structure that stores the state of the random number generator so that each thread generates its own sequence of random numbers.

		curand_init(seed + id, 0, 0, &state); // initializes random # generator state for each thread
		//seed + id is to make sure the random #s are different for each thread.
		//0 is the sequence # and allows for the generation of multiple random # sequences.
		//I set it to 0 because I only need 1 random # sequence per thread.
		//2nd 0 is the offset and specifies the starting point of the random # sequence. It is set to 0 so it starts at the beginning.
		//&state passes the adress of curandState to the curand_init function so it can update the state of the random # generator.

		int initialPosition = 0; // initializing starting pos to 0

		for (int i = 0; i < numberOfRandomSteps; i++) // loops through #ofrandomsteps. each iteration is 1 step in the random walk.
		{
			unsigned int randomNumber = curand(&state); //generates random unsigned int from 0 to UINT_MAX. state makes sure each thread genertes its own random # sequence.
			int step = (randomNumber < midPoint) ? -1 : 1; //if random# < midpoint is TRUE step is -1 if its false step = 1.
			initialPosition += step; //updating current position by adding step value -1 or 1
		}
		finalPositions[id] = initialPosition; //store the final position for each thread.
	}
}
/*
 RAND_MAX = 2147483647
 rand() returns a value in [0, 2147483647].
 Because RAND_MAX is odd and we are also using 0 this is an even number.
 Hence there is no middle interger so RAND_MAX/2 will divide the number in half if it is a float.
 You might could do this faster with a clever idea using ints but I'm going to use a float.
 Also I'm not sure how long the string of random numbers is. I'm sure it is longer than 10,000.
 Before you use this as a huge string check this out.
*/
// int getRandomDirection()
// {	
// 	int randomNumber = rand();
	
// 	if(randomNumber < MidPoint) return(-1);
// 	else return(1);
// }

int main(int argc, char** argv)
{
	setupDevices(); //calling setupDevices function

	int numberOfRandomSteps = STEPS; // initializing # oof steps for each walk


	//launching randomwalk kernal on GPU.
	//time(NULL) generates a seed for the random # generator bases on the current time to make sure the random walks are different with each run.
	randomWalk<<<GridSize, BlockSize>>>(GPUfinalPositions, time(NULL), numberOfRandomSteps, MidPoint); // Launch kernel
	cudaErrorCheck(__FILE__, __LINE__); 
	cudaDeviceSynchronize(); // Synchronize to ensure all threads have finished before copying results back to CPU
	cudaErrorCheck(__FILE__, __LINE__); 

	//copying the results of random walks from GPU to CPU
	cudaMemcpy(CPUfinalPositions, GPUfinalPositions, WALKS * sizeof(int), cudaMemcpyDeviceToHost); // Copy results back to CPU
	cudaErrorCheck(__FILE__, __LINE__);
	
	
	printf(" Final positions of the random walks:\n");
	for(int i = 0; i < WALKS; i++) //loops through CPUfinalPositions array to print the final pos of each walk.
	{
		printf("Walk %d: %d\n", i, CPUfinalPositions[i]); // Print final positions of each walk
	}
	
	cudaFree(GPUfinalPositions); // Free memory allocated on GPU for GPUfinalPositions array. This prevents memory leaks.
	
	return 0; //returns 0 to OS to indicate a successful run.
}

