
// Name:
// nvcc HW6.cu -o temp -lglut -lGL
// glut and GL are openGL libraries.
/*
 What to do:
 This code displays a simple Julia fractal using the CPU.
 Rewrite the code so that it uses the GPU to create the fractal. 
 Keep the window at 1024 by 1024.
*/

// Include files
#include <stdio.h>
#include <GL/glut.h>

// Defines
#define MAXMAG 10.0 // If you grow larger than this, we assume that you have escaped.
#define MAXITERATIONS 200 // If you have not escaped after this many attempts, we assume you are not going to escape.
#define A  -0.824	//Real part of C
#define B  -0.1711	//Imaginary part of C

void setUpDevices()
{
	BlockSize.x = 1024;
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	GridSize.x = 1024; 
	GridSize.y = 1;
	GridSize.z = 1;
}
// Global variables

float *CPUpixels;
float *GPUpixels;
unsigned int WindowWidth = 1024;
unsigned int WindowHeight = 1024;

float XMin = -2.0;
float XMax =  2.0;
float YMin = -2.0;
float YMax =  2.0;

// Function prototypes
void cudaErrorCheck(const char*, int);
__global__ void escapeOrNotColorGPU(float*, float, float, float, float, unsigned int, unsigned int);

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

void allocateMemory()
{
	float4 CPUpixels = (float*)malloc(WindowWidth*WindowHeight*3*sizeof(float));
	cudaMalloc(&GPUpixels,WindowWidth*WindowHeight*3*sizeof(float));
}

void initializePixels()
{
	for(int i=0; i < WindowWidth*WindowHeight*3; i++)
	{
		CPUpixels[i].x = 0.0;
		CPUpixels[i].y = 0.0;
		CPUpixels[i].z = 0.0;
		CPUpixels[i].w = 1.0; //black
	}
}

__global__
void escapeOrNotColorGPU(float *pixels, float XMin, float XMax, float YMin, float YMax, unsigned int WindowWidth, unsigned int WindowHeight)
{
	
 int id = blockIdx.x * blockDim.x + threadIdx.x;

    float x, y, stepSizeX, stepSizeY;
        float mag, tempX;
        int count;

        stepSizeX = (XMax - XMin)/((float)WindowWidth);
        stepSizeY = (YMax - YMin)/((float)WindowHeight);
        
        y = YMin + stepSizeY*threadIdx.x;
        x = XMin + stepSizeX*blockIdx.x;
        
        count = 0;
        mag = sqrt(x*x + y*y);
        while (mag < MAXMAG && count < MAXITERATIONS) 
        {	
            tempX = x; //We will be changing the x but we need its old value to find y.
            x = x*x - y*y + A;
            y = (2.0 * tempX * y) + B;
            mag = sqrt(x*x + y*y);
            count++;
        }
        
		int pixelIndex = (blockIdx.x * blockDim.x + threadIdx.x) * 3;
		if (count < MAXITERATIONS) {
			pixels[id].x = 0.0; // Red
			pixels[id].y = 0.0; // Green
			pixels[id].z = 0.0; // Blue
			pixels[id].w = 1.0; // Alpha
		} else {
			pixels[id].x = 1.0; // Red
			pixels[id].y = 0.0; // Green
			pixels[id].z = 0.0; // Blue
			pixels[id].w = 1.0; // Alpha
		}
    }
}

/*float escapeOrNotColor (float x, float y) 
{
	float mag,tempX;
	int count;
	
	int maxCount = MAXITERATIONS;
	float maxMag = MAXMAG;
	
	count = 0;
	mag = sqrt(x*x + y*y);;
	while (mag < maxMag && count < maxCount) 
	{	
		tempX = x; //We will be changing the x but we need its old value to find y.
		x = x*x - y*y + A;
		y = (2.0 * tempX * y) + B;
		mag = sqrt(x*x + y*y);
		count++;
	}
	if(count < maxCount) 
	{
		return(0.0);
	}
	else
	{
		return(1.0);
	}
}
*/
void display(void) 
{ 

escapeOrNotColorGPU<<<GridSize, BlockSize>>>(Pixels_GPU, N, XMax, XMin, YMax, YMin, WindowWidth, WindowHeight);
	cudaErrorCheck(__FILE__, __LINE__);

	//copy the pixels from the GPU to the CPU
	cudaMemcpy(Pixels_CPU, Pixels_GPU, cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);

	//Putting pixels on the screen.
	glDrawPixels(WindowWidth, WindowHeight, GL_RGBA, GL_FLOAT, Pixels_CPU);
	glFlush();

}

void cleanUp()
{
	free(Pixels_CPU);
	cudaFree(Pixels_GPU);
	printf("Memory freed. Exiting...\n");

	exit(0);
}

int main(int argc, char** argv)
{ 
	setUpDevices();


   	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
   	glutInitWindowSize(WindowWidth, WindowHeight);
	glutCreateWindow("Fractals--Man--Fractals");
   	glutDisplayFunc(display);
   	glutMainLoop();

	allocateMemory();
	initializePixels();
	cudaMemcpy(Pixels_GPU, Pixels_CPU, cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);

	cleanup();
	return 0;
}
