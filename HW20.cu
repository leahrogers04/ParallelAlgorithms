// Name:
// Optimizing nBody GPU code. 
// nvcc HW20.cu -o temp -lglut -lm -lGLU -lGL

/*
 What to do:
 This is some lean n-body code that runs on the GPU for any number of bodies (within reason). Take this code and make it 
 run as fast as possible using any tricks you know or can find. Try to keep the same general format so we can time it and 
 compare it with others' code. This will be a competition. To focus more on new ideas rather than just using a bunch of if 
 statements to avoid going out of bounds, N will be a power of 2 and 256 < N < 262,144. CHeck in code to make sure this is true.
 Note: The code takes two arguments as inputs:
 1. The number of bodies to simulate.
 2. Whether to draw sub-arrangements of the bodies during the simulation (1), or only the first and last arrangements (0).
*/

// Include files
#include <GL/glut.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// Defines
#define BLOCK_SIZE 256
#define PI 3.14159265359
#define DRAW_RATE 10

// This is to create a Lennard-Jones type function G/(r^p) - H(r^q). (p < q) p has to be less than q.
// In this code we will keep it a p = 2 and q = 4 problem. The diameter of a body is found using the general
// case so it will be more robust but in the code leaving it as a set 2, 4 problem make the coding much easier.
#define G 10.0f
#define H 10.0f
#define LJP  2.0
#define LJQ  4.0

#define DT 0.0001
#define RUN_TIME 1.0

// Globals
int N, DrawFlag;
float4 *P, *V, *F;
float *M; 
float4 *PGPU, *VGPU, *FGPU;
float *MGPU;
float GlobeRadius, Diameter, Radius;
float Damp;
dim3 BlockSize;
dim3 GridSize;

// Function prototypes
void cudaErrorCheck(const char *, int);
void keyPressed(unsigned char, int, int);
long elaspedTime(struct timeval, struct timeval);
void drawPicture();
void timer();
void setup();
__global__ void getForces(float4 *, float4 *, float4 *, float *, float, float, int);
__global__ void moveBodies(float4 *, float4 *, float4 *, float *, float, float, float, int);
void nBody();
int main(int, char**);

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

void keyPressed(unsigned char key, int x, int y)
{
	if(key == 's')
	{
		printf("\n The simulation is running.\n");
		timer();
	}
	
	if(key == 'q')
	{
		exit(0);
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

void drawPicture()
{
	int i;
	
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	
	cudaMemcpyAsync(P, PGPU, N*sizeof(float4), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);
	
	glColor3d(1.0,1.0,0.5);
	for(i=0; i<N; i++)
	{
		glPushMatrix();
		glTranslatef(P[i].x, P[i].y, P[i].z);
		glutSolidSphere(Radius,20,20);
		glPopMatrix();
	}
	
	glutSwapBuffers();
}

void timer()
{	
	timeval start, end;
	long computeTime;
	
	drawPicture();
	gettimeofday(&start, NULL);
    		nBody();
    		cudaDeviceSynchronize();
		cudaErrorCheck(__FILE__, __LINE__);
    	gettimeofday(&end, NULL);
    	drawPicture();
    	
	computeTime = elaspedTime(start, end);
	printf("\n The compute time was %ld microseconds.\n\n", computeTime);
}

void setup()
{
    	float randomAngle1, randomAngle2, randomRadius;
    	float d, dx, dy, dz;
    	int test;
    	
    	BlockSize.x = BLOCK_SIZE;
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	GridSize.x = (N - 1)/BlockSize.x + 1; //Makes enough blocks to deal with the whole vector.
	GridSize.y = 1;
	GridSize.z = 1;
	
    	Damp = 0.5;
    	
    	M = (float*)malloc(N*sizeof(float));
    	P = (float4*)malloc(N*sizeof(float4));
    	V = (float4*)malloc(N*sizeof(float4));
    	F = (float4*)malloc(N*sizeof(float4));
    	
    	cudaMalloc(&MGPU,N*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&PGPU,N*sizeof(float4));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&VGPU,N*sizeof(float4));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&FGPU,N*sizeof(float4));
	cudaErrorCheck(__FILE__, __LINE__);
    	
	Diameter = pow(H/G, 1.0/(LJQ - LJP)); // This is the value where the force is zero for the L-J type force.
	Radius = Diameter/2.0;
	
	// Using the radius of a body and a 68% packing ratio to find the radius of a global sphere that should hold all the bodies.
	// Then we double this radius just so we can get all the bodies setup with no problems. 
	float totalVolume = float(N)*(4.0/3.0)*PI*Radius*Radius*Radius;
	totalVolume /= 0.68;
	float totalRadius = pow(3.0*totalVolume/(4.0*PI), 1.0/3.0);
	GlobeRadius = 2.0*totalRadius;
	
	// Randomly setting these bodies in the glaobal sphere and setting the initial velosity, inotial force, and mass.
	for(int i = 0; i < N; i++)
	{
		test = 0;
		while(test == 0)
		{
			// Get random position.
			randomAngle1 = ((float)rand()/(float)RAND_MAX)*2.0*PI;
			randomAngle2 = ((float)rand()/(float)RAND_MAX)*PI;
			randomRadius = ((float)rand()/(float)RAND_MAX)*GlobeRadius;
			P[i].x = randomRadius*cos(randomAngle1)*sin(randomAngle2);
			P[i].y = randomRadius*sin(randomAngle1)*sin(randomAngle2);
			P[i].z = randomRadius*cos(randomAngle2);
			
			// Making sure the balls centers are at least a diameter apart.
			// If they are not throw these positions away and try again.
			test = 1;
			for(int j = 0; j < i; j++)
			{
				dx = P[i].x-P[j].x;
				dy = P[i].y-P[j].y;
				dz = P[i].z-P[j].z;
				d = sqrt(dx*dx + dy*dy + dz*dz);
				if(d < Diameter)
				{
					test = 0;
					break;
				}
			}
		}
	
		
		V[i].x = 0.0;
		V[i].y = 0.0;
		V[i].z = 0.0;
		
		F[i].x = 0.0;
		F[i].y = 0.0;
		F[i].z = 0.0;
		
		M[i] = 1.0;
	}
	
	cudaMemcpyAsync(PGPU, P, N*sizeof(float4), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(VGPU, V, N*sizeof(float4), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(FGPU, F, N*sizeof(float4), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(MGPU, M, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	
	printf("\n To start timing type s.\n");
}


// __global__ void getForces(float3 *p, float3 *v, float3 *f, float *m, float g, float h, int n)
// {
// 	float dx, dy, dz,d,d2;
// 	float force_mag;
	
// 	int i = threadIdx.x + blockDim.x*blockIdx.x;
	
// 	if(i < n)
// 	{
// 		f[i].x = 0.0f;
// 		f[i].y = 0.0f;
// 		f[i].z = 0.0f;
		
// 		for(int j = 0; j < n; j++)
// 		{
// 			if(i != j)
// 			{
// 				dx = p[j].x-p[i].x;
// 				dy = p[j].y-p[i].y;
// 				dz = p[j].z-p[i].z;
// 				d2 = dx*dx + dy*dy + dz*dz;
// 				d  = sqrt(d2);
				
// 				force_mag  = (g*m[i]*m[j])/(d2) - (h*m[i]*m[j])/(d2*d2);
// 				f[i].x += force_mag*dx/d;
// 				f[i].y += force_mag*dy/d;
// 				f[i].z += force_mag*dz/d;
// 			}
// 		}
// 	}
// }

// __global__ void moveBodies(float3 *p, float3 *v, float3 *f, float *m, float damp, float dt, float t, int n)
// {	
// 	int i = threadIdx.x + blockDim.x*blockIdx.x;
	
// 	if(i < n)
// 	{
// 		if(t == 0.0f)
// 		{
// 			v[i].x += ((f[i].x-damp*v[i].x)/m[i])*dt/2.0f;
// 			v[i].y += ((f[i].y-damp*v[i].y)/m[i])*dt/2.0f;
// 			v[i].z += ((f[i].z-damp*v[i].z)/m[i])*dt/2.0f;
// 		}
// 		else
// 		{
// 			v[i].x += ((f[i].x-damp*v[i].x)/m[i])*dt;
// 			v[i].y += ((f[i].y-damp*v[i].y)/m[i])*dt;
// 			v[i].z += ((f[i].z-damp*v[i].z)/m[i])*dt;
// 		}

// 		p[i].x += v[i].x*dt;
// 		p[i].y += v[i].y*dt;
// 		p[i].z += v[i].z*dt;
// 	}
// }

__global__ void getForces(float4 *p, float4 *f, float *m, float g, float h, int n) {
    __shared__ float4 sharedP[BLOCK_SIZE]; //shared mem arrays to make it faster
    __shared__ float sharedM[BLOCK_SIZE];

    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;

    if (id >= n) return; //making sure that threads outside the range of the number of bodies isnt doing anything

    float4 myPos = p[id]; //position of current body
    float4 force = {0.0f, 0.0f, 0.0f}; //initializing force to zero

    for (int tile = 0; tile < (n + BLOCK_SIZE - 1) / BLOCK_SIZE; tile++) //the tile is used to divide it into smaller chunks that fit in shared memory. it optimizes memory access by reusing data in shared mem
	 {
        
        int idx = tile * BLOCK_SIZE + tid;
			if (idx < n)  //puts the pos and masses in the current tile into shared mem
			{
				sharedP[tid] = p[idx];
				sharedM[tid] = m[idx];
			}
			 else
			{ 
				sharedP[tid] = {0.0f, 0.0f, 0.0f};
				sharedM[tid] = 0.0f;
			}
        __syncthreads();

        #pragma unroll 4 //unrol loop to make faster
        for (int j = 0; j < BLOCK_SIZE; j++) 
		{
            if (tile * BLOCK_SIZE + j >= n || id == tile * BLOCK_SIZE + j) continue;

            float dx = sharedP[j].x - myPos.x;
            float dy = sharedP[j].y - myPos.y;
            float dz = sharedP[j].z - myPos.z;
            float d2 = dx * dx + dy * dy + dz * dz + 1e-6f; // Avoid division by zero
            float d = sqrtf(d2);

            float force_mag = (g * m[id] * sharedM[j]) / d2 - (h * m[id] * sharedM[j]) / (d2 * d2);
            force.x += force_mag * dx / d;
            force.y += force_mag * dy / d;
            force.z += force_mag * dz / d;
        }
        __syncthreads();
    }

  
    f[id] = force; //putting the net force back into global mem
}

__global__ void moveBodies(float4 *p, float4 *v, float4 *f, float *m, float damp, float dt, int n) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= n) return;

	//float invMass = invM[id]; //inverse mass so its faster
    // Update velocity
    v[id].x += ((f[id].x - damp * v[id].x)/m[id]) * dt;
    v[id].y += ((f[id].y - damp * v[id].y)/m[id]) * dt;
    v[id].z += ((f[id].z - damp * v[id].z)/m[id]) * dt;

    // Update position
    p[id].x += v[id].x * dt;
    p[id].y += v[id].y * dt;
    p[id].z += v[id].z * dt;
}

void nBody()
 {
    int drawCount = 0;
    float t = 0.0f;
    float dt = 0.0001f;

    while (t < RUN_TIME) {
        // Compute forces
        getForces<<<GridSize, BlockSize>>>(PGPU, FGPU, MGPU, G, H, N);
        cudaErrorCheck(__FILE__, __LINE__);

        // Move bodies
        moveBodies<<<GridSize, BlockSize>>>(PGPU, VGPU, FGPU, MGPU, Damp, dt, N);
        cudaErrorCheck(__FILE__, __LINE__);

        // Draw the simulation at regular intervals
        if (drawCount == DRAW_RATE) {
            if (DrawFlag) {
                drawPicture();
            }
            drawCount = 0;
        }

        t += dt;
        drawCount++;
    }
}
// void nBody()
// {
// 	int    drawCount = 0; 
// 	float  t = 0.0;
// 	float dt = 0.0001;

// 	while(t < RUN_TIME)
// 	{
// 		getForces<<<GridSize,BlockSize>>>(PGPU, VGPU, FGPU, MGPU, G, H, N);
// 		cudaErrorCheck(__FILE__, __LINE__);
// 		moveBodies<<<GridSize,BlockSize>>>(PGPU, VGPU, FGPU, MGPU, Damp, dt, t, N);
// 		cudaErrorCheck(__FILE__, __LINE__);
// 		if(drawCount == DRAW_RATE) 
// 		{
// 			if(DrawFlag) 
// 			{	
// 				drawPicture();
// 			}
// 			drawCount = 0;
// 		}
		
// 		t += dt;
// 		drawCount++;
// 	}
// }

int main(int argc, char** argv)
{
	if( argc < 3)
	{
		printf("\n You need to enter the number of bodies (an int)"); 
		printf("\n and if you want to draw the bodies as they move (1 draw, 0 don't draw),");
		printf("\n on the comand line.\n"); 
		exit(0);
	}
	else
	{
		N = atoi(argv[1]);
		DrawFlag = atoi(argv[2]);
	}
	
	setup();
	
	int XWindowSize = 1000;
	int YWindowSize = 1000;
	
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGB);
	glutInitWindowSize(XWindowSize,YWindowSize);
	glutInitWindowPosition(0,0);
	glutCreateWindow("nBody Test");
	GLfloat light_position[] = {1.0, 1.0, 1.0, 0.0};
	GLfloat light_ambient[]  = {0.0, 0.0, 0.0, 1.0};
	GLfloat light_diffuse[]  = {1.0, 1.0, 1.0, 1.0};
	GLfloat light_specular[] = {1.0, 1.0, 1.0, 1.0};
	GLfloat lmodel_ambient[] = {0.2, 0.2, 0.2, 1.0};
	GLfloat mat_specular[]   = {1.0, 1.0, 1.0, 1.0};
	GLfloat mat_shininess[]  = {10.0};
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glShadeModel(GL_SMOOTH);
	glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);
	glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient);
	glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_COLOR_MATERIAL);
	glEnable(GL_DEPTH_TEST);
	glutKeyboardFunc(keyPressed);
	glutDisplayFunc(drawPicture);
	
	float3 eye = {0.0f, 0.0f, 2.0f*GlobeRadius};
	float near = 0.2;
	float far = 5.0*GlobeRadius;
	
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glFrustum(-0.2, 0.2, -0.2, 0.2, near, far);
	glMatrixMode(GL_MODELVIEW);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	gluLookAt(eye.x, eye.y, eye.z, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
	
	glutMainLoop();
	return 0;
}





