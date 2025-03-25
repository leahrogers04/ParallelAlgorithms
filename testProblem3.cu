#include <sys/time.h>
#include <stdio.h>


#define N 1001 // Length of  vector
#define B 200
#define BLOCK_SIZE 10

float *A_CPU, *B_CPU, *C_CPU; 
float *A_GPU, *B_GPU, *C_GPU; 
float DotCPU, DotGPU;
dim3 BlockSize; //This variable will hold the Dimensions of your blocks
dim3 GridSize; //This variable will hold the Dimensions of your grid
float Tolerance = 0.01;

// Function prototypes
void cudaErrorCheck(const char *, int);
void setUpDevices();
void allocateMemory();
void innitialize();
void dotProductCPU(float*, float*, int);
__global__ void dotProductGPU(float*, float*, float*, int);
bool  check(float, float, float);
long elaspedTime(struct timeval, struct timeval);
void cleanUp();

// This check to see if an error happened in your CUDA code. It tell you what it thinks went wrong,
// and what file and line it occured on.
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

// This will be the layout of the parallel space we will be using.
void setUpDevices()
{
   
    BlockSize.x = BLOCK_SIZE;
    BlockSize.y = 1;
    BlockSize.z = 1;
   
    GridSize.x = 4;
    GridSize.y = 1;
    GridSize.z = 1;
}

// Allocating the memory we will be using.
void allocateMemory()
{  
    // Host "CPU" memory.              
    A_CPU = (float*)malloc(N*sizeof(float));
    B_CPU = (float*)malloc(N*sizeof(float));
    C_CPU = (float*)malloc(N*sizeof(float));
   
    // Device "GPU" Memory
    cudaMalloc(&A_GPU,N*sizeof(float));
    cudaErrorCheck(__FILE__, __LINE__);
    cudaMalloc(&B_GPU,N*sizeof(float));
    cudaErrorCheck(__FILE__, __LINE__);
    cudaMalloc(&C_GPU,N*sizeof(float));
    cudaErrorCheck(__FILE__, __LINE__);
}

// Loading values into the vectors that we will add.
void innitialize()
{
    for(int i = 0; i < N; i++)
    {      
        A_CPU[i] = (float)i;    
        B_CPU[i] = (float)(3*i);
    }
}

// Adding vectors a and b on the CPU then stores result in vector c.
void dotProductCPU(float *a, float *b, float *C_CPU, int n)
{
    for(int id = 0; id < n; id++)
    {
        C_CPU[id] = a[id] * b[id];
    }
   
    for(int id = 1; id < n; id++)
    {
        C_CPU[0] += C_CPU[id];
    }
}

// This is the kernel. It is the function that will run on the GPU.
// It adds vectors a and b on the GPU then stores result in vector c.
__global__ void dotProductGPU(float *a, float *b, float *c, int n)
{
   
    int tid = threadIdx.x;
    __shared__ float dot[BLOCK_SIZE];
    int id;
    int stride = blockDim.x*gridDim.x; //total number of threads in the grid. This is like the step size for how much the threads should step over the array for each blocks iteration
    int maxIteration = ((n-1)/stride)   +1; // this formula accounts for the last elements that might not fill up an entire block
    int fold;
    for (int i=0; i<=maxIteration;i++) //this loops through all the strides and each iteration is for a set of elements that a block will work on
    {
        id = tid+stride*i +blockDim.x*blockIdx.x;
            if (i < maxIteration) //checks if the current iteration is less than the last one and if it is it multiplies a and b and stores it in dot
            {
                dot[tid] = a[id]*b[id];
           
                fold = blockDim.x;
                while(fold>1) //this loop reduces or folds the values in the shared mem array dot so that the end result is a single float
                                //it will keep going until the only value left is dot[0]
                {
                    if(tid<fold/2)
                    {
                        dot[tid] += dot[fold-1-tid]; //adding the value from the end of the block. thread 0 will be added to the last thread and so on
                    }
                    fold = (fold+1)/2; // updating fold and adding 1 so that it is always an odd number. fold will keep getting cut in half until it equals 1 which will hold the final sum dot
                }
        }
    else if (i == maxIteration) //if the id is still in the bounds of the array then it will calc the dot product for the remaining elements
                                //but if id is bigger than n then the thread stores 0's in dot so that all the threads still get processed.
    {
        if(id<n)
        {
            dot[tid] = a[id]*b[id];
        }
        else
        {
            dot[tid] = 0;//padding with zeroes
        }
        fold = blockDim.x; //resetting fold so that it does it all again with the new number of blocks
        while(fold>1)
        {
            if(tid<fold/2)
            {
                dot[tid] += dot[fold-1-tid];
            }
            fold = (fold+1)/2;
        }
    }
    if(tid==0)
    {
    atomicAdd(&c[0],dot[tid]);
    }
}
}






// Checking to see if anything went wrong in the vector addition.
bool check(float cpuAnswer, float gpuAnswer, float tolerence)
{
    double percentError;
   
    percentError = abs((gpuAnswer - cpuAnswer)/(cpuAnswer))*100.0;
    printf("\n\n percent error = %lf\n", percentError);
   
    if(percentError < Tolerance)
    {
        return(true);
    }
    else
    {
        return(false);
    }
}

// Calculating elasped time.
long elaspedTime(struct timeval start, struct timeval end)
{
    // tv_sec = number of seconds past the Unix epoch 01/01/1970
    // tv_usec = number of microseconds past the current second.
   
    long startTime = start.tv_sec * 1000000 + start.tv_usec; // In microseconds.
    long endTime = end.tv_sec * 1000000 + end.tv_usec; // In microseconds

    // Returning the total time elasped in microseconds
    return endTime - startTime;
}

// Cleaning up memory after we are finished.
void CleanUp()
{
    // Freeing host "CPU" memory.
    free(A_CPU);
    free(B_CPU);
    free(C_CPU);
   
    cudaFree(A_GPU);
    cudaErrorCheck(__FILE__, __LINE__);
    cudaFree(B_GPU);
    cudaErrorCheck(__FILE__, __LINE__);
    cudaFree(C_GPU);
    cudaErrorCheck(__FILE__, __LINE__);
}

int main()
{
    int DotGPU = 0;
    timeval start, end;
    long timeCPU, timeGPU;
    //float localC_CPU, localC_GPU;
   
    // Setting up the GPU
    setUpDevices();
   
    // Allocating the memory you will need.
    allocateMemory();
   
    // Putting values in the vectors.
    innitialize();
   
    // Adding on the CPU
    gettimeofday(&start, NULL);
    dotProductCPU(A_CPU, B_CPU, C_CPU, N);
    DotCPU = C_CPU[0];
    gettimeofday(&end, NULL);
    timeCPU = elaspedTime(start, end);
   
   
   
    // Adding on the GPU
    gettimeofday(&start, NULL);
   
    // Copy Memory from CPU to GPU      
    cudaMemcpyAsync(A_GPU, A_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaErrorCheck(__FILE__, __LINE__);
    cudaMemcpyAsync(B_GPU, B_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaErrorCheck(__FILE__, __LINE__);
   
    dotProductGPU<<<GridSize,BlockSize>>>(A_GPU, B_GPU, C_GPU, N);
    cudaErrorCheck(__FILE__, __LINE__);
   
    // Copy Memory from GPU to CPU  
    cudaMemcpyAsync(C_CPU, C_GPU, sizeof(float), cudaMemcpyDeviceToHost);
    cudaErrorCheck(__FILE__, __LINE__);
   
    DotGPU = C_CPU[0]; // C_GPU was copied into C_CPU.
   
    // Making sure the GPU and CPU wiat until each other are at the same place.
    cudaDeviceSynchronize();
    cudaErrorCheck(__FILE__, __LINE__);

    gettimeofday(&end, NULL);
    timeGPU = elaspedTime(start, end);
   
    // Checking to see if all went correctly.
    if(check(DotCPU, DotGPU, Tolerance) == false)
    {
        printf("\n\n Something went wrong in the GPU dot product.\n");
    }
    else
    {
        printf("\n\n You did a dot product correctly on the GPU");
        printf("\n The time it took on the CPU was %ld microseconds", timeCPU);
        printf("\n The time it took on the GPU was %ld microseconds", timeGPU);
    }
   
    // Your done so cleanup your room.  
    CleanUp();  
   
    // Making sure it flushes out anything in the print buffer.
    printf("\n\n");
   
    return(0);
}
