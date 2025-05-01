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
float MidPoint = (float)UINT_MAX/2.0f;
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

	//allocating memory
	CPUfinalPositions = (int *)malloc(WALKS * sizeof(int));
	cudaMalloc(&GPUfinalPositions, WALKS * sizeof(int)); // Allocate memory on the GPU for final positions
	cudaErrorCheck(__FILE__, __LINE__); // Check for errors after memory allocation
}

__global__ void randomWalk(int *finalPositions, unsigned long long seed, int numberOfRandomSteps, float midPoint)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;

	if (id < WALKS)
	{
		curandState state;
		curand_init(seed + id, 0, 0, &state);

		int initialPosition = 0;

		for (int i = 0; i < numberOfRandomSteps; i++)
		{
			unsigned int randomNumber = curand(&state);
			int step = (randomNumber < midPoint) ? -1 : 1; //curand() returns a random number in [0, 1)
			initialPosition += step; //updating current position with each step
		}
		finalPositions[id] = initialPosition; //store the final position in the array
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
	setupDevices(); 

	int numberOfRandomSteps = STEPS;

	randomWalk<<<GridSize, BlockSize>>>(GPUfinalPositions, time(NULL), numberOfRandomSteps, MidPoint); // Launch kernel
	cudaErrorCheck(__FILE__, __LINE__); 
	cudaDeviceSynchronize();
	cudaErrorCheck(__FILE__, __LINE__); 

	//copying the results back to the CPU
	cudaMemcpy(CPUfinalPositions, GPUfinalPositions, WALKS * sizeof(int), cudaMemcpyDeviceToHost); // Copy results back to CPU
	cudaErrorCheck(__FILE__, __LINE__);
	
	
	printf(" Final positions of the random walks:\n");
	for(int i = 0; i < WALKS; i++)
	{
		printf("Walk %d: %d\n", i, CPUfinalPositions[i]); // Print final positions of each walk
	}
	
	cudaFree(GPUfinalPositions); // Free GPU memory
	
	return 0;
}

