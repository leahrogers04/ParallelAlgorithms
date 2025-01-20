// Name: Leah Rogers
// nvcc HW1.cu -o temp
/*
 What to do:
 1. Understand every line of code and be able to explain it in class.
 2. Compile, run, and play around with the code.
*/

// Include files
#include <sys/time.h> //prepocessor directive that includes functions to measure time
#include <stdio.h>  //prepocessor directive that includes input output operations like printf

// Defines
#define N 100000 // Length of the vector
// using #define makes it easy to change the value of N in one place without having to change it everywhere in the code.

// Global variables
float *A_CPU, *B_CPU, *C_CPU; // 3 global pointers to float arrays that are used to allocate and store the vectors in the CPU's memory
float Tolerance = 0.00000001; // tolerance is because floating point numbers arent always exact so we need to check if the numbers are close enough to each other to be considered equal

// Function prototypes
// function prototypes are used to tell the compiler that the function will be used later
void allocateMemory(); // telling the compiler that the function will return no value
void innitialize();
void addVectorsCPU(float*, float*, float*, int);
int  check(float*, int);
long elaspedTime(struct timeval, struct timeval);
void CleanUp();

//Allocating the memory we will be using.
void allocateMemory() //this is the function definition or whats indside the function
{	
	// Host "CPU" memory.	

	//allocated memory for  vector A_CPU.
	// malloc allocates N elements of type float and returns a pointer to the allocated memory.
	// (float*) converts the pointer returned by malloc to a float pointer			
	A_CPU = (float*)malloc(N*sizeof(float));
	B_CPU = (float*)malloc(N*sizeof(float));
	C_CPU = (float*)malloc(N*sizeof(float));
}

//Loading values into the vectors that we will add.
void innitialize() // function definition
				  // initializes the A and B CPU arrays
{
	for(int i = 0; i < N; i++) // loop that goes from 0 to N-1
	{		
		A_CPU[i] = (float)i; // assigns the value of i which is cast to a float to the ith element in the A_CPU array	
		B_CPU[i] = (float)(2*i); // assigns the value of 2*i which is cast to a float to the ith element in the B_CPU array
	}
}

//Adding vectors a and b then stores result in vector c.
void addVectorsCPU(float *a, float *b, float *c, int n) // float *a means that a is a pointer to a float array
														// int n is the number of elements in the arrays
{
	for(int id = 0; id < n; id++) // loop iterating from 0 to n-1.
								  // when id reaches n-1 the body executes with id=n-1 and after the body id is incremented to n. 
								  //the condition id < n is checked again but it is now false since id is equal to n and the loop terminates.
	{ 
		c[id] = a[id] + b[id]; // adds the ith element of a to the ith element of b and stores the result in the ith element of c
	}
}

// Checking to see if anything went wrong in the vector addition.
int check(float *c, int n) // defines function that returns an int
{
	int id; // declares interger variable id
	double sum = 0.0; // declares a double sum and initializes it to 0.0. sum will accumilate the sum of elements in c
	double m = n-1; // Needed the -1 because we start at 0.
	
	for(id = 0; id < n; id++) // loop that iterates from 0 to n-1. id starts at 0 and increments by 1 until it reaches n .
	{ 
		sum += c[id]; // adds the idth element of c to sum
	}
	
	if(abs(sum - 3.0*(m*(m+1))/2.0) < Tolerance) // checks if the absolute value of the difference between sum and 3.0*(m*(m+1))/2.0 is less than Tolerance
	{
		return(1); // if the difference is less that tolerance the function returns 1. this indicates that the sum is correct
	}
	else 
	{
		return(0); // if the difference is greater than tolerance the function returns 0. this indicates that the sum is incorrect
	}
}

// Calculating elasped time.
long elaspedTime(struct timeval start, struct timeval end)
{
	// tv_sec = number of seconds past the Unix epoch 01/01/1970
	// tv_usec = number of microseconds past the current second.
	
	long startTime = start.tv_sec * 1000000 + start.tv_usec; // In microseconds.
	//how long its been since 1970 when the program starts
	long endTime = end.tv_sec * 1000000 + end.tv_usec; // In microseconds
	//how long its been since 1970 when the program ends
	

	// Returning the total time elasped in microseconds
	return endTime - startTime; //// Subtracting the start time from the end time to get total elapsed time
}

//Cleaning up memory after we are finished.
void CleanUp() // this function is important because it prevents memory leaks that happen when allocated memory is not properly deallocated.
{
	// Freeing host "CPU" memory.
	free(A_CPU); // the free function deallocates memory that was previously allocated by malloc (freeing the memory)
	free(B_CPU); 
	free(C_CPU);
}

int main()
{
	timeval start, end; // declares 2 variables of type timeval that will be used to store the start and end times for the elapsed time measured
	
	// Allocating the memory you will need.
	allocateMemory(); // calling the function allocateMemory to allocate memory for the vectors A_CPU, B_CPU, and C_CPU
	
	// Putting values in the vectors.
	innitialize(); // calling the function innitialize to put values in the vectors A_CPU and B_CPU

	// Starting the timer.	
	gettimeofday(&start, NULL); // gettimeofday function is used to get the current time. the first argument is a pointer to a timeval struct and the second argument is NULL

	// Add the two vectors.
	addVectorsCPU(A_CPU, B_CPU ,C_CPU, N); // calling the function addVectorsCPU to add the vectors A_CPU and B_CPU and store the result in C_CPU

	// Stopping the timer.
	gettimeofday(&end, NULL); // gettimeofday function is used to get the current time. the first argument is a pointer to a timeval struct and the second argument is NULL which indicates that no time info is being provides. Null means that the pointer doesnt point to anything
	
	// Checking to see if all went correctly.
	if(check(C_CPU, N) == 0)
	{
		printf("\n\n Something went wrong in the vector addition\n"); // if the check function returns 0 then something went wrong in the vector addition
	}
	else
	{
		printf("\n\n You added the two vectors correctly on the CPU"); // if the check function returns 1 then the vectors were added correctly
		printf("\n The time it took was %ld microseconds", elaspedTime(start, end));
	}
	
	// Your done so cleanup your room.	
	CleanUp();	// calling the function CleanUp to deallocate the memory that was allocated by the allocateMemory function
	
	// Making sure it flushes out anything in the print buffer.
	printf("\n");
	
	return(0); // returns 0 to indicate that the program ran successfully
}

