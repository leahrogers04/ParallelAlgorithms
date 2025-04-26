// Name:
// CPU random walk. 
// nvcc HW27.cu -o temp

/*
 What to do:
 Create a function that returns a random number that is either -1 or 1.
 Start at 0 and call this function to move you left (-1) or right (1) one step each call.
 Do this 10000 times and print out your final position.
*/

// Include files


#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Defines
#define STEPS 10000 // Number of steps in the random walk

// Globals


// Function prototypes

//int randomWalk();
//int main(int, char**);

int randomWalk()
{
	int initialPosition = 0; //starting at 0

	for(int i = 0; i < STEPS; i++)
	{
		int step = (rand() % 2 == 0) ? -1 : 1; //rand() % 2 makes sure that the number is either 0 or 1. 
											  //and then it checks if the result is zero ( even number)
		                                      //if this is tue it returns -1 otherwise it returns 1
		initialPosition += step; //updating current position with each step
		//printf("initial position at Step %d: %d\n", i, initialPosition); // uncomment this line to see the position at each step
		//printf("Step %d: %d\n", i, step); // uncomment this line to see the step taken at each step
	}
	return initialPosition;
}

int main(int argc, char** argv)
{
	srand(time(NULL)); // seeding the random number generator 
					  //using time(NULL) returns the current time since the beginning of computer time (1/1/1970 i think)
					  //using this as the seed ensures that the random numbers are different each time you run the program.
					  //because the seed changes with the current time.

	//int steps = 10000; // decided to make it a #define since we are gonna be expanding the code to GPU.
	int finalPosition = randomWalk();

	printf("Final Position after %d steps: %d\n", STEPS, finalPosition);
	return 0;
}

