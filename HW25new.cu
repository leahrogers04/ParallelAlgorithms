// Name:
// nBody run on all available GPUs. 
// nvcc HW25.cu -o temp -lglut -lm -lGLU -lGL

/*
 What to do:
 This is some robust N-body code with all the bells and whistles removed. 
 It runs on two GPUs and two GPUs only. Rewrite it so it automatically detects the number of 
 available GPUs on the machine and runs using all of them.
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
#define RUN_TIME 1.0

// Globals
int N;
//int HalfN; // Half the vector size
int bodiesPerGPU;
int NumberOfGpus;
float3 *P, *V, *F;
float *M; 
float3 **PGPU, **VGPU, **FGPU;
float **MGPU;
int *NGPU; //number of bodie  on each gpu
float GlobeRadius, Diameter, Radius;
float Damp;
dim3 BlockSize;
dim3 GridSize;
int *offsetGPU;


// Function prototypes
void cudaErrorCheck(const char *, int);
void drawPicture();
void setup();
__global__ void getForces(float3 *, float3 *, float3 *, float *, float, float, int, int, int);
__global__ void moveBodies(float3 *, float3 *, float3 *, float *, float, float, float, int, int, int);
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
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	
	cudaSetDevice(0);
	cudaMemcpyAsync(P, PGPU[0], N*sizeof(float3), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);
	
	glColor3d(1.0,1.0,0.5);
	
	for(int i=0; i<N; i++)
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
    	float randomAngle1, randomAngle2, randomRadius;
    	float d, dx, dy, dz;
    	int test;
	
	N = 101;
	
	cudaGetDeviceCount(&NumberOfGpus);
	if (NumberOfGpus < 1)
	{
		printf("\n Dude, you don't even have a GPU. Sorry, you can't play with us. Call NVIDIA and buy a GPU — loser!\n");
		exit(0);
	}
		PGPU = (float3**)malloc(NumberOfGpus*sizeof(float3*));
		VGPU = (float3**)malloc(NumberOfGpus*sizeof(float3*));	
		FGPU = (float3**)malloc(NumberOfGpus*sizeof(float3*));
		MGPU = (float**)malloc(NumberOfGpus*sizeof(float*));
		NGPU = (int*)malloc(NumberOfGpus*sizeof(int));
		offsetGPU = (int*)malloc(NumberOfGpus*sizeof(int));

		if(N % NumberOfGpus != 0)
		{
			N+= (NumberOfGpus - (N % NumberOfGpus));
		}
		bodiesPerGPU = N/NumberOfGpus;

		for(int i = 0; i < NumberOfGpus; i++)
		{
			PGPU[i] = (float3*)malloc(bodiesPerGPU*sizeof(float3));
			VGPU[i] = (float3*)malloc(bodiesPerGPU*sizeof(float3));
			FGPU[i] = (float3*)malloc(bodiesPerGPU*sizeof(float3));
			MGPU[i] = (float*)malloc(bodiesPerGPU*sizeof(float));
			NGPU[i] = bodiesPerGPU;
			offsetGPU[i] = i*bodiesPerGPU;
			if(i == NumberOfGpus - 1)
			{
				NGPU[i] += N % NumberOfGpus;
			}
		}
	}
		
		BlockSize.x = BLOCK_SIZE;
		BlockSize.y = 1;
		BlockSize.z = 1;
		
		GridSize.x = (bodiesPerGPU + BlockSize.x - 1)/BlockSize.x; // This gives us the correct number of blocks.
		GridSize.y = 1;
		GridSize.z = 1;

		for(int i = 0; i < NumberOfGpus; i++)
		{
			cudaSetDevice(i);
			cudaMalloc((void**)&PGPU[i], N*sizeof(float3));
			cudaErrorCheck(__FILE__, __LINE__);
			cudaMalloc((void**)&VGPU[i], N*sizeof(float3));
			cudaErrorCheck(__FILE__, __LINE__);
			cudaMalloc((void**)&FGPU[i], N*sizeof(float3));
			cudaErrorCheck(__FILE__, __LINE__);
			cudaMalloc((void**)&MGPU[i], N*sizeof(float));
			cudaErrorCheck(__FILE__, __LINE__);
		}
	
    	Damp = 0.5;
    	
    	M = (float*)malloc(N*sizeof(float));
    	P = (float3*)malloc(N*sizeof(float3));
    	V = (float3*)malloc(N*sizeof(float3));
    	F = (float3*)malloc(N*sizeof(float3));
    	
    	// !! Important: Setting the number of bodies a little bigger if it is not even or you will 
    	// get a core dump because you will be copying memory you do not own. This only needs to be
    	// done for positions but I did it for all for completness incase the code gets used for a
    	// more complicated force function.
    	
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
			
			// Making sure the bodies' centers are at least a diameter apart.
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
	
	for (int i = 0; i < numberOfGpus; i++)
	{
	cudaSetDevice(i);
	cudaMemcpyAsync(PGPU[i], P, N*sizeof(float3), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(VGPU[i], V, N*sizeof(float3), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(FGPU[i], F, N*sizeof(float3), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(MGPU[i], M, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	}
		
	printf("\n Setup finished.\n");
}

__global__ void getForces(float3 *p, float3 *v, float3 *f, float *m, float g, float h, int n, int bodiesPerGpu)
{
	float dx, dy, dz,d,d2;
	float force_mag;
	int offset;
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	int id = i + offset;
	
	if(i < bodiesPerGPU)
	{
		f[i].x = 0.0f;
		f[i].y = 0.0f;
		f[i].z = 0.0f;
		
		for(int j = 0; j < n; j++)
		{
			if(id != j)
			{
				dx = p[j].x-p[id].x;
				dy = p[j].y-p[id].y;
				dz = p[j].z-p[id].z;
				d2 = dx*dx + dy*dy + dz*dz;
				d  = sqrt(d2);
				
				force_mag  = (g*m[id]*m[j])/(d2) - (h*m[id]*m[j])/(d2*d2);
				f[i].x += force_mag*dx/d;
				f[i].y += force_mag*dy/d;
				f[i].z += force_mag*dz/d;
			}
		}
	}
}

__global__ void moveBodies(float3 *p, float3 *v, float3 *f, float *m, float damp, float dt, float t, int n,)
{
	int offset;
	int i = threadIdx.x + blockDim.x*blockIdx.x + offset;
	int id = i + offset;
	
	if(i < n)
	{
		if(t == 0.0f)
		{
			v[i].x += ((f[i].x-damp*v[i].x)/m[id])*dt/2.0f;
			v[i].y += ((f[i].y-damp*v[i].y)/m[id])*dt/2.0f;
			v[i].z += ((f[i].z-damp*v[i].z)/m[id])*dt/2.0f;
		}
		else
		{
			v[i].x += ((f[i].x-damp*v[i].x)/m[id])*dt;
			v[i].y += ((f[i].y-damp*v[i].y)/m[id])*dt;
			v[i].z += ((f[i].z-damp*v[i].z)/m[id])*dt;
		}

		p[id].x += v[i].x*dt;
		p[id].y += v[i].y*dt;
		p[id].z += v[i].z*dt;
	}
}

void nBody()
{
	int    drawCount = 0; 
	float  t = 0.0;
	float dt = 0.0001;

	while(t < RUN_TIME)
	{
		for(int i = 0; i < NumberOfGpus; i++)
		{
			cudaSetDevice(i);
			getForces<<<GridSize,BlockSize>>>(PGPU[i], VGPU[i], FGPU[i], MGPU[i], G, H, N, bodiesPerGPU, OffsetGPU[i]);
			cudaErrorCheck(__FILE__, __LINE__);
			moveBodies<<<GridSize,BlockSize>>>(PGPU[i], VGPU[i], FGPU[i], MGPU[i], Damp, dt, t, N, bodiesPerGPU, OffsetGPU[i]);
			cudaErrorCheck(__FILE__, __LINE__);
        }
        for(int i = 0; i < NumberOfGpus; i++)
        {
            cudaSetDevice(i);
            cudaDeviceSynchronize();
            cudaErrorCheck(__FILE__, __LINE__);
		}

		if (NumberOfGpus > 1)
		{
			for( int dest = 0; dest < NumberOfGpus; dest++)
			{
				for(int gpu = 0; gpu < NumberOfGpus; gpu++)
				{
					if(gpu != dest)
					{
						cudaSetDevice(dest);
						cudaMemcpyAsync(PGPU[dest] + OffsetGPU[gpu], PGPU[gpu] + OffsetGPU[gpu], N*sizeof(float3), cudaMemcpyDeviceToDevice);
						cudaErrorCheck(__FILE__, __LINE__);
					}
				}
			}
		}
	
		// for (int i = 0; i < NumberOfGpus; i++)
		// {
		// 	cudaSetDevice(i);
		// 	cudaDeviceSynchronize();
		// 	cudaErrorCheck(__FILE__, __LINE__);
		// }
		

		if(drawCount == DRAW_RATE) 
		{	
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

