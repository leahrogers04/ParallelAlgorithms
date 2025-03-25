// Name: Leah Rogers
// Two body problem
// nvcc HW17.cu -o temp -lglut -lGLU -lGL
//To stop hit "control c" in the window you launched it from.

/*
 What to do:
 This is some crude code that moves two bodies around in a box, attracted by gravity and 
 repelled when they hit each other. Take this from a two-body problem to an N-body problem, where 
 NUMBER_OF_SPHERES is a #define that you can change. Also clean it up a bit so it is more user-friendly.
*/

// Include files
#include <GL/glut.h>
#include <GL/glu.h>
#include <GL/gl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Defines
#define NUMBER_OF_SPHERES 20
#define XWindowSize 1000
#define YWindowSize 1000
#define STOP_TIME 10000.0
#define DT        0.0001
#define GRAVITY 0.1 
#define MASS 10.0  	
#define DIAMETER 1.0
#define SPHERE_PUSH_BACK_STRENGTH 50.0
#define PUSH_BACK_REDUCTION 0.1
#define DAMP 0.01
#define DRAW 100
#define LENGTH_OF_BOX 6.0
#define MAX_VELOCITY 5.0

// Globals
const float XMax = (LENGTH_OF_BOX/2.0);
const float YMax = (LENGTH_OF_BOX/2.0);
const float ZMax = (LENGTH_OF_BOX/2.0);
const float XMin = -(LENGTH_OF_BOX/2.0);
const float YMin = -(LENGTH_OF_BOX/2.0);
const float ZMin = -(LENGTH_OF_BOX/2.0);
//float px1, py1, pz1, vx1, vy1, vz1, fx1, fy1, fz1, mass1; 
//float px2, py2, pz2, vx2, vy2, vz2, fx2, fy2, fz2, mass2;

//arrays storing the values for N spheres
//float px[NUMBER_OF_SPHERES], py[NUMBER_OF_SPHERES], pz[NUMBER_OF_SPHERES];
//float vx[NUMBER_OF_SPHERES], vy[NUMBER_OF_SPHERES], vz[NUMBER_OF_SPHERES];
//float fx[NUMBER_OF_SPHERES], fy[NUMBER_OF_SPHERES], fz[NUMBER_OF_SPHERES];
//float mass[NUMBER_OF_SPHERES];

//struct is better actually because itll be easier to read and less error prone
struct spheresStruct 
{
	float px, py, pz, vx, vy, vz, fx, fy, fz, mass;
};

	spheresStruct *sphere;

// Function prototypes
void set_initail_conditions();
void Drawwirebox();
void draw_picture();
void keep_in_box();
void get_forces();
void move_bodies(float);
void nbody();
void Display(void);
void reshape(int, int);
int main(int, char**);

void set_initail_conditions()
{ 
	time_t t;
	srand((unsigned) time(&t));
	//int yeahBuddy;
	float dx, dy, dz, seperation;

	for (int i = 0; i < NUMBER_OF_SPHERES; i++)
	{
		if (i == 0)
		{
			sphere[i].px = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
			sphere[i].py = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
			sphere[i].pz = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
		}
		else
		{
			sphere[i].px = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
			sphere[i].py = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
			sphere[i].pz = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;

			//bool goodPosition = true;
		
		//bool goodPosition = false;
		//yeahBuddy = 0; // Make sure the spheres are not on top of each other.
		for (int j = 0; j < i; j++)
		{
			bool goodPosition = false;//true;
			while(goodPosition)
			{

				goodPosition = false;
				dx = sphere[j].px - sphere[i].px;
				dy = sphere[j].py - sphere[i].py;
				dz = sphere[j].pz - sphere[i].pz;
				seperation = sqrt(dx*dx + dy*dy + dz*dz);
				if(seperation < DIAMETER)
				{
					goodPosition = true;//false;
					break;
				}
			}
				dx = sphere[j].px - sphere[i].px;
				dy = sphere[j].py - sphere[i].py;
				dz = sphere[j].pz - sphere[i].pz;
				seperation = sqrt(dx*dx + dy*dy + dz*dz);
				if(seperation < DIAMETER)
				{
					goodPosition = true;
					break;
				}
		}
	}
	
	/*float speed = MAX_VELOCITY * 0.5; // Set a fixed initial speed
	float theta = 2.0 * M_PI * rand() / RAND_MAX; // Random angle in the XY plane
	float phi = acos(2.0 * rand() / RAND_MAX - 1.0); // Random angle from Z-axis

	sphere[i].vx = speed * sin(phi) * cos(theta);
	sphere[i].vy = speed * sin(phi) * sin(theta);
	sphere[i].vz = speed * cos(phi);
	*/
		sphere[i].vx = MAX_VELOCITY * (rand() / (float)RAND_MAX - 0.5) * 0.5;
        sphere[i].vy = MAX_VELOCITY * (rand() / (float)RAND_MAX - 0.5) * 0.5;
        sphere[i].vz = MAX_VELOCITY * (rand() / (float)RAND_MAX - 0.5) * 0.5;

	//sphere[i].vx = 2.0*MAX_VELOCITY*rand()/RAND_MAX - MAX_VELOCITY;
	//sphere[i].vy = 2.0*MAX_VELOCITY*rand()/RAND_MAX - MAX_VELOCITY;
	//sphere[i].vz = 2.0*MAX_VELOCITY*rand()/RAND_MAX - MAX_VELOCITY;
	printf("Sphere %d: vx = %f, vy = %f, vz = %f\n", i, sphere[i].vx, sphere[i].vy, sphere[i].vz);
	
	sphere[i].mass = MASS;
	}
}
	


void Drawwirebox()
{		
	glColor3f (5.0,1.0,1.0);
	glBegin(GL_LINE_STRIP);
		glVertex3f(XMax,YMax,ZMax);
		glVertex3f(XMax,YMax,ZMin);	
		glVertex3f(XMax,YMin,ZMin);
		glVertex3f(XMax,YMin,ZMax);
		glVertex3f(XMax,YMax,ZMax);
		
		glVertex3f(XMin,YMax,ZMax);
		
		glVertex3f(XMin,YMax,ZMax);
		glVertex3f(XMin,YMax,ZMin);	
		glVertex3f(XMin,YMin,ZMin);
		glVertex3f(XMin,YMin,ZMax);
		glVertex3f(XMin,YMax,ZMax);	
	glEnd();
	
	glBegin(GL_LINES);
		glVertex3f(XMin,YMin,ZMax);
		glVertex3f(XMax,YMin,ZMax);		
	glEnd();
	
	glBegin(GL_LINES);
		glVertex3f(XMin,YMin,ZMin);
		glVertex3f(XMax,YMin,ZMin);		
	glEnd();
	
	glBegin(GL_LINES);
		glVertex3f(XMin,YMax,ZMin);
		glVertex3f(XMax,YMax,ZMin);		
	glEnd();
	
}

void draw_picture()
{
	float radius = DIAMETER/2.0;
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	
	Drawwirebox();
	
	for (int i = 0; i < NUMBER_OF_SPHERES; i++)
	{
		glColor3d(1.0,0.5,1.0);
		glPushMatrix();
		glTranslatef(sphere[i].px, sphere[i].py, sphere[i].pz);
		glutSolidSphere(radius,20,20);
		glPopMatrix();
	}

	
	glutSwapBuffers();
}

void keep_in_box()
{
	float halfBoxLength = (LENGTH_OF_BOX - DIAMETER)/2.0;
	
	for(int i = 0; i < NUMBER_OF_SPHERES; i++)
	{
		if(sphere[i].px > halfBoxLength)
		{
			sphere[i].px = 2.0*halfBoxLength - sphere[i].px;
			sphere[i].vx = -sphere[i].vx;
		}
		else if (sphere[i].px < -halfBoxLength)
		{
			sphere[i].px = -2.0*halfBoxLength - sphere[i].px;
			sphere[i].vx = -sphere[i].vx;
		}
		if(sphere[i].py > halfBoxLength)
		{
			sphere[i].py = 2.0*halfBoxLength - sphere[i].py;
			sphere[i].vy = -sphere[i].vy;
		}
		else if (sphere[i].py < -halfBoxLength)
		{
			sphere[i].py = -2.0*halfBoxLength - sphere[i].py;
			sphere[i].vy = -sphere[i].vy;
		}
		if(sphere[i].pz > halfBoxLength)
		{
			sphere[i].pz = 2.0*halfBoxLength - sphere[i].pz;
			sphere[i].vz = -sphere[i].vz;
		}
		else if (sphere[i].pz < -halfBoxLength)
		{
			sphere[i].pz = -2.0*halfBoxLength - sphere[i].pz;
			sphere[i].vz = -sphere[i].vz;
		}
	}
}

void get_forces()
{
	float epsilon = 0.0001;
	float dx,dy,dz,r,r2,dvx,dvy,dvz,forceMag,inout;
	const float maxForce = 1.0;
	//setting forces to 0
	for (int i = 0; i< NUMBER_OF_SPHERES; i++)
	{
		sphere[i].fx = 0.0;
		sphere[i].fy = 0.0;
		sphere[i].fz = 0.0;
	}

	printf("sphere %d: fx = %f, fy = %f, fz = %f\n", 0, sphere[0].fx, sphere[0].fy, sphere[0].fz);
	
	for(int i = 0; i < NUMBER_OF_SPHERES; i++)
	{
		for(int j = i + 1; j < NUMBER_OF_SPHERES; j++)
		{
			dx = sphere[j].px - sphere[i].px;
			dy = sphere[j].py - sphere[i].py;
			dz = sphere[j].pz - sphere[i].pz;
			r2 = dx*dx + dy*dy + dz*dz + epsilon;
			r = sqrt(r2);
			
			forceMag =  sphere[i].mass * sphere[j].mass * GRAVITY / r2;
			
			if (r < DIAMETER)
			{
				dvx = sphere[j].vx - sphere[i].vx;
				dvy = sphere[j].vy - sphere[i].vy;
				dvz = sphere[j].vz - sphere[i].vz;
				inout = dx*dvx + dy*dvy + dz*dvz;
				if(inout <= 0.0)
				{
					forceMag +=  SPHERE_PUSH_BACK_STRENGTH * (r - DIAMETER);
				}
				else
				{
					forceMag +=  PUSH_BACK_REDUCTION * SPHERE_PUSH_BACK_STRENGTH * (r - DIAMETER);
				}
			}

			if(forceMag > maxForce)
			{
				forceMag = maxForce;
			}
			
			sphere[i].fx += forceMag * dx / r;
			sphere[i].fy += forceMag * dy / r;
			sphere[i].fz += forceMag * dz / r;
			sphere[j].fx -= forceMag * dx / r;
			sphere[j].fy -= forceMag * dy / r;
			sphere[j].fz -= forceMag * dz / r;
		}
	}
}
				
// Collision resolution function
void resolve_collision(spheresStruct *sphere1, spheresStruct *sphere2)
{
    float dx = sphere2->px - sphere1->px;
    float dy = sphere2->py - sphere1->py;
    float dz = sphere2->pz - sphere1->pz;

    float distance = sqrt(dx * dx + dy * dy + dz * dz);

    // Check if the spheres are overlapping
    if (distance < DIAMETER)
    {
		  // Avoid division by zero or very small numbers
		  if (distance == 0.0f)
		  {
			  distance = 1e-6; // Assign a small value to prevent division by zero
		  }
        // Normalize the collision vector
        float nx = dx / distance;
        float ny = dy / distance;
        float nz = dz / distance;

        // Relative velocity
        float dvx = sphere2->vx - sphere1->vx;
        float dvy = sphere2->vy - sphere1->vy;
        float dvz = sphere2->vz - sphere1->vz;

        // Velocity along the collision normal
        float vn = dvx * nx + dvy * ny + dvz * nz;

        // If spheres are moving toward each other
        if (vn < 0)
        {
            // Elastic collision response
            float m1 = sphere1->mass;
            float m2 = sphere2->mass;

            float impulse = (2 * vn) / (m1 + m2);

			const float k = 0.2;
			impulse *= k;

			const float MAX_IMPULSE = 1.0;
            if (impulse > MAX_IMPULSE)
            {
                impulse = MAX_IMPULSE;
            }
            else if (impulse < -MAX_IMPULSE)
            {
                impulse = -MAX_IMPULSE;
            }


           // Separate the spheres to prevent overlap
			float overlap = DIAMETER - distance;
			float correctionFactor = 0.5; // Adjust this factor to control the separation speed

			//this is the only thing that is making the balls stay calm if you take it out the balls will start going crazy and all hell will break loose beware

			sphere1->px -= correctionFactor * 0.5 * overlap * nx;
			sphere1->py -= correctionFactor * 0.5 * overlap * ny;
			sphere1->pz -= correctionFactor * 0.5 * overlap * nz;

			sphere2->px += correctionFactor * 0.5 * overlap * nx;
			sphere2->py += correctionFactor * 0.5 * overlap * ny;
			sphere2->pz += correctionFactor * 0.5 * overlap * nz;
			printf("Collision: Sphere1 vx = %f, vy = %f, vz = %f | Sphere2 vx = %f, vy = %f, vz = %f | impulse = %f\n",
       sphere1->vx, sphere1->vy, sphere1->vz, sphere2->vx, sphere2->vy, sphere2->vz, impulse);
        }
		

        // Separate the spheres to prevent overlap
        float overlap = DIAMETER - distance;
        sphere1->px -= 0.5 * overlap * nx;
        sphere1->py -= 0.5 * overlap * ny;
        sphere1->pz -= 0.5 * overlap * nz;

        sphere2->px += 0.5 * overlap * nx;
        sphere2->py += 0.5 * overlap * ny;
        sphere2->pz += 0.5 * overlap * nz;
    }
}

void move_bodies(float time)
{
	for (int i = 0; i < NUMBER_OF_SPHERES; i++)
	{
		if(time == 0.0)
		{
			sphere[i].vx += 0.5 * DT * (sphere[i].fx - DAMP * sphere[i].vx) / sphere[i].mass;
			sphere[i].vy += 0.5 * DT * (sphere[i].fy - DAMP * sphere[i].vy) / sphere[i].mass;
			sphere[i].vz += 0.5 * DT * (sphere[i].fz - DAMP * sphere[i].vz) / sphere[i].mass;
		}
		else
		{
			sphere[i].vx += DT * (sphere[i].fx - DAMP * sphere[i].vx) / sphere[i].mass;
			sphere[i].vy += DT * (sphere[i].fy - DAMP * sphere[i].vy) / sphere[i].mass;
			sphere[i].vz += DT * (sphere[i].fz - DAMP * sphere[i].vz) / sphere[i].mass;
		}

		sphere[i].px += DT * sphere[i].vx;
		sphere[i].py += DT * sphere[i].vy;
		sphere[i].pz += DT * sphere[i].vz;
	}

	    // Check for collisions between spheres
		for (int k = 0; k <2; k++)
		{
			for (int i = 0; i < NUMBER_OF_SPHERES; i++)
			{
				for (int j = i + 1; j < NUMBER_OF_SPHERES; j++)
				{
					resolve_collision(&sphere[i], &sphere[j]);
				}
			}
 	}

	keep_in_box();
}

void nbody()
{	
	int    tdraw = 0;
	float  time = 0.0;

	sphere = (spheresStruct *)malloc(NUMBER_OF_SPHERES * sizeof(spheresStruct));
		if (sphere == NULL) 
		{
    printf("Error: Memory allocation failed.\n");
    exit(1);
	}
	set_initail_conditions();
	
	draw_picture();
	
	while(time < STOP_TIME)
	{
		get_forces();
	
		move_bodies(time);
	
		tdraw++;
		if(tdraw == DRAW) 
		{
			draw_picture(); 
			tdraw = 0;
		}
		
		time += DT;
	}
	printf("\n DONE \n");
	while(1);

	free(sphere);
}

void Display(void)
{
	gluLookAt(0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glutSwapBuffers();
	glFlush();
	nbody();
}

void reshape(int w, int h)
{
	glViewport(0, 0, (GLsizei) w, (GLsizei) h);

	glMatrixMode(GL_PROJECTION);

	glLoadIdentity();

	glFrustum(-0.2, 0.2, -0.2, 0.2, 0.2, 50.0);

	glMatrixMode(GL_MODELVIEW);
}

int main(int argc, char** argv)
{
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGB);
	glutInitWindowSize(XWindowSize,YWindowSize);
	glutInitWindowPosition(0,0);
	glutCreateWindow("2 Body 3D");
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
	glutDisplayFunc(Display);
	glutReshapeFunc(reshape);
	glutMainLoop();
	return 0;
}


