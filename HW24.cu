// Name:
// nBody code on multiple GPUs. 
// nvcc HW24.cu -o temp -lglut -lm -lGLU -lGL

/*
 What to do:
 This is some robust N-body code with all the bells and whistles removed. 
 Modify it so it runs on two GPUs.
*/

// Include files
#include <GL/glut.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// Defines
#define BLOCK_SIZE 128
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
#define RUN_TIME 10.0

// Globals
int N, N1, N2;
float3 *P, *V, *F;
float *M; 
float3 *PGPU1, *VGPU1, *FGPU1;
float3 *PGPU2, *VGPU2, *FGPU2;
float *MGPU1, *MGPU2;
float GlobeRadius, Diameter, Radius;
float Damp;
dim3 BlockSize;
dim3 GridSize1;
dim3 GridSize2;
//int halfN = (N+1)/2; // This is the half of the vector length
//int otherHalfN = N - halfN; // This is the other half of the vector length


// Function prototypes
void cudaErrorCheck(const char *, int);
void drawPicture();
void setup();
__global__ void getForces(float3 *, float3 *, float3 *, float *, float, float, int);
__global__ void moveBodies(float3 *, float3 *, float3 *, float *, float, float, float, int);
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

void drawPicture()
{
	int i;
	
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	
	// cudaSetDevice(0); // Set the first GPU
	// cudaMemcpyAsync(PGPU1, P, N1*sizeof(float3), cudaMemcpyHostToDevice);	
	// cudaErrorCheck(__FILE__, __LINE__);

	// cudaSetDevice(1); // Set the second GPU
	// cudaMemcpyAsync(PGPU2, P, N2*sizeof(float3), cudaMemcpyDeviceToHost);
	// cudaErrorCheck(__FILE__, __LINE__);
	
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

void setup()
{
	int count = 0;
	cudaGetDeviceCount(&count);
	if (count < 2)
	{
		printf("you only have %d 2 GPU's stupid idiot loser \n", count);
		exit (0);

	}
    	float randomAngle1, randomAngle2, randomRadius;
    	float d, dx, dy, dz;
    	int test;
    	
    	N = 1000;
		N1 = N/2; // This is the half of the vector length
		N2 = N - N1; // This is the other half of the vector length
		
		// float H = 10.0f;
		// float G = 10.0f;
		
		// float LJP = 2.0;
		// float LJQ = 4.0;
		
		float GlobeRadius, Diameter, Radius;
		
		// This is the layout of the parallel space we will be using.
		BlockSize.x = BLOCK_SIZE;
    	
    BlockSize.x = BLOCK_SIZE;
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	GridSize1.x = (N1 - 1) / BlockSize.x + 1; //Makes enough blocks to deal with the whole vector.
	GridSize1.y = 1;
	GridSize1.z = 1;
	
	GridSize2.x = (N2 - 1) / BlockSize.x + 1; //Makes enough blocks to deal with the whole vector.
	GridSize2.y = 1;
	GridSize2.z = 1;
	

    	Damp = 0.5;
    	
    	M = (float*)malloc(N*sizeof(float));
    	P = (float3*)malloc(N*sizeof(float3));
    	V = (float3*)malloc(N*sizeof(float3));
    	F = (float3*)malloc(N*sizeof(float3));
    	
	cudaSetDevice(0); // Set the first GPU
	cudaMalloc(&PGPU1,N*sizeof(float3));			
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&VGPU1,N*sizeof(float3));	
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&FGPU1,N*sizeof(float3));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&MGPU1,N*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
    
	cudaSetDevice(1); // Set the second GPU
	cudaMalloc(&PGPU2,N*sizeof(float3));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&VGPU2,N*sizeof(float3));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&FGPU2,N*sizeof(float3));
	cudaErrorCheck(__FILE__, __LINE__);	
	cudaMalloc(&MGPU2,N*sizeof(float));
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
	cudaSetDevice(0); // Set the first GPU
	cudaMemcpyAsync(PGPU1, P, N1*sizeof(float3), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(VGPU1, V, N1*sizeof(float3), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(FGPU1, F, N1*sizeof(float3), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(MGPU1, M, N1*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);

	cudaSetDevice(1); // Set the second GPU
	cudaMemcpyAsync(PGPU2, P, N2*sizeof(float3), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(VGPU2, V, N2*sizeof(float3), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(FGPU2, F+N1, N2*sizeof(float3), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(MGPU2, M+N1, N2*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);

// 	int canAccessPeer01, canAccessPeer10;
// cudaDeviceCanAccessPeer(&canAccessPeer01, 0, 1);
// cudaDeviceCanAccessPeer(&canAccessPeer10, 1, 0);

// if (canAccessPeer01) {
//     cudaSetDevice(0);
//     cudaDeviceEnablePeerAccess(1, 0);
// }

// if (canAccessPeer10) {
//     cudaSetDevice(1);
//     cudaDeviceEnablePeerAccess(0, 0);
// }
 }

__global__ void getForces(float3 *p, float3 *v, float3 *f, float *m, float g, float h, int n)
{
	float dx, dy, dz,d,d2;
	float force_mag;
	
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	
	if(i < n)
	{
		f[i].x = 0.0f;
		f[i].y = 0.0f;
		f[i].z = 0.0f;
		
		for(int j = 0; j < n; j++)
		{
			if(i != j)
			{
				dx = p[j].x-p[i].x;
				dy = p[j].y-p[i].y;
				dz = p[j].z-p[i].z;
				d2 = dx*dx + dy*dy + dz*dz;
				d  = sqrt(d2);
				
				force_mag  = (g*m[i]*m[j])/(d2) - (h*m[i]*m[j])/(d2*d2);
				f[i].x += force_mag*dx/d;
				f[i].y += force_mag*dy/d;
				f[i].z += force_mag*dz/d;
			}
		}
	}
}

__global__ void moveBodies(float3 *p, float3 *v, float3 *f, float *m, float damp, float dt, float t, int n)
{	
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	
	if(i < n)
	{
		if(t == 0.0f)
		{
			v[i].x += ((f[i].x-damp*v[i].x)/m[i])*dt/2.0f;
			v[i].y += ((f[i].y-damp*v[i].y)/m[i])*dt/2.0f;
			v[i].z += ((f[i].z-damp*v[i].z)/m[i])*dt/2.0f;
		}
		else
		{
			v[i].x += ((f[i].x-damp*v[i].x)/m[i])*dt;
			v[i].y += ((f[i].y-damp*v[i].y)/m[i])*dt;
			v[i].z += ((f[i].z-damp*v[i].z)/m[i])*dt;
		}

		p[i].x += v[i].x*dt;
		p[i].y += v[i].y*dt;
		p[i].z += v[i].z*dt;
	}
}

void nBody()
{
	int    drawCount = 0; 
	float  t = 0.0;
	float dt = 0.0001;

	while(t < RUN_TIME)
	{
		cudaSetDevice(0); // Set the first GPU
		getForces<<<GridSize1,BlockSize>>>(PGPU1, VGPU1, FGPU1, MGPU1, G, H, N1);
		cudaErrorCheck(__FILE__, __LINE__);
		moveBodies<<<GridSize1,BlockSize>>>(PGPU1, VGPU1, FGPU1, MGPU1, Damp, dt, t, N1);
		cudaErrorCheck(__FILE__, __LINE__);

		cudaSetDevice(1); // Set the second GPU
		getForces<<<GridSize2,BlockSize>>>(PGPU2, VGPU2, FGPU2, MGPU2, G, H, N2);
		cudaErrorCheck(__FILE__, __LINE__);
		moveBodies<<<GridSize2,BlockSize>>>(PGPU2, VGPU2, FGPU2, MGPU2, Damp, dt, t, N2);
		cudaErrorCheck(__FILE__, __LINE__);

		cudaDeviceSynchronize(); // Wait for the GPU to finish before moving on.

		cudaSetDevice(0); // Set the first GPU	
		cudaDeviceSynchronize(); // Wait for the GPU to finish before moving on.

		cudaSetDevice(0); // Set the first GPU
		cudaMemcpyAsync(PGPU2, PGPU1, N1*sizeof(float3), cudaMemcpyDeviceToDevice);
		cudaErrorCheck(__FILE__, __LINE__);

		cudaSetDevice(1); // Set the second GPU
		cudaMemcpyAsync(PGPU1+N1, PGPU2+N1, N2*sizeof(float3), cudaMemcpyDeviceToDevice);
		cudaErrorCheck(__FILE__, __LINE__);

		cudaSetDevice(0); // Set the first GPU
		cudaDeviceSynchronize(); // Wait for the GPU to finish before moving on.
		cudaSetDevice(1); // Set the second GPU	
		cudaDeviceSynchronize(); // Wait for the GPU to finish before moving on.
		if(drawCount == DRAW_RATE) 
		{	
			cudaSetDevice(0); // Set the first GPU
			cudaMemcpyAsync(P, PGPU1, N1*sizeof(float3), cudaMemcpyDeviceToHost);
			cudaErrorCheck(__FILE__, __LINE__);

			cudaSetDevice(1); // Set the second GPU
			cudaMemcpyAsync(P+N1,PGPU2+N1, N2*sizeof(float3), cudaMemcpyDeviceToHost);
			cudaErrorCheck(__FILE__, __LINE__);

			cudaDeviceSynchronize(); // Wait for the GPU to finish before moving on.
			cudaSetDevice(0); // Set the first GPU
			cudaDeviceSynchronize(); // Wait for the GPU to finish before moving on.
			drawPicture();
			drawCount = 0;
		}
		
		t += dt;
		drawCount++;
	}
}

int main(int argc, char** argv)
{
	setup();
	
	int XWindowSize = 1000;
	int YWindowSize = 1000;
	
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGB);
	glutInitWindowSize(XWindowSize,YWindowSize);
	glutInitWindowPosition(0,0);
	glutCreateWindow("Nbody Two GPUs");
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
	glutDisplayFunc(drawPicture);
	glutIdleFunc(nBody);
	
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

