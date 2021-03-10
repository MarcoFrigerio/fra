
#include <pyopencl-complex.h>
#define MAX_ITERATIONS 30
#define MANDELBROT_THRESHOLD 2
#define OUTPUT_SIZE_IN_PIXELS 2000
#define RGB 3

typedef struct {
  uint r;
  uint g;
  uint b;
  uint iters;
  float coeff;
} rgb;

rgb julia_iterations(
	cfloat_t C,
	cfloat_t j,
	float rot ,
	uint input_i,
	float input_thre)
	{
		uint iterations = 1;
		rgb rrgb;
		float iterfloat=iterations;
		float speed0=0;
		float speed;
		// cfloat_t one;
		// one.x=20/iterations;
		// one.y=0;
		// one=cfloat_add(j,one);
		cfloat_t Z = C;
		#pragma unroll
		//while (iterations < MAX_ITERATIONS){
		while (iterations < input_i){
			//Z = cfloat_divide(cfloat_mul(Z,Z),cfloat_cos(Z));
			Z=cfloat_powr(Z,2);
			//Z=cfloat_tan(Z);
			//Z = cfloat_mul(Z,Z);
			//Z = cfloat_mul(Z,Z);
			//j=cfloat_powr(j,rot);
			//j = cfloat_mul(j,cfloat_powr(j,rot));
			//Z = cfloat_cos(Z,cfloat_e(j));
			Z = cfloat_add(Z,j);
			speed+=cfloat_abs(Z)-speed0;
			speed0=cfloat_abs(Z);
			//Z = cfloat_divide(Z,cfloat_tan(j));
			if(cfloat_abs(Z) > input_thre){break;}
			iterations++;
			iterfloat=iterations;
			}
		
		float coeff=(iterfloat/input_i)*128;
		float absz = cfloat_abs(Z);
		float r;float g; float b;
		if (coeff<128){
			// r=fmax(coeff*absz,255);
			// g=fmax(coeff*2,255);
			// b=fmax((2-absz*coeff/2),255);
			r=coeff*absz;
			g=Z.y*Z.y;
			b=255-(absz)*48;
			}
		else{
			// r=fmax(absz*32,255);
			// g=fmax(255-128*absz,255);
			// b=fmax(absz*48,255);
			r=100-(speed*100);//native_cos(absz)*255;
			g=speed;
			b=speed*50;
			}
		rrgb.r=r;
		rrgb.g=g;
		rrgb.b=b;
		rrgb.iters=iterations;
		rrgb.coeff=coeff;
		// if (coeff<128){rrgb.r=coeff*absz;rrgb.g=absz*50;rrgb.b=(2-absz)*128;}
		// else{rrgb.r=absz*256;rrgb.g=absz*128;rrgb.b=(2-absz)*128;}
		// // // rrgb.r=coeff;
		// rrgb.g=absz*50;
		// rrgb.b=(2-absz)*128;
		return rrgb;
	}

__kernel void julia(
	__global float * gD,
	const unsigned int input_i,
	const float input_thre,
	__global uint * result
	)
{

	uint gid_x = get_global_id(0);
	uint gid_y = get_global_id(1);
	uint gid_z = get_global_id(2);


	// if (gid_x==1000 && gid_y==1000 && gid_z==1 && input_i==1)
	// {
	// 	uint wkdim=get_work_dim ();
	// 	size_t ls0=get_global_offset(0);
	// 	size_t ls1=get_global_offset(1);
	// 	size_t ls2=get_global_offset(2);

	// 	printf("wkdim %d \n",wkdim);
	// 	printf("offset 0 %d \n",ls0);
	// 	printf("offset 1 %d \n",ls1);
	// 	printf("offset 2 %d \n",ls2);
	// }


	float x = gD[gid_x];
	float y = gD[gid_y];
	
	float finput_i = input_i;
	finput_i=cos(finput_i) ;
	uint l_input_i=input_i;
	//uint l_input_i=2*finput_i+input_i;
	// if (gid_x==1000 && gid_y==1000 && gid_z==1 )
	// 	{printf("input_i %d finput_i  %4.4f l_input_i %d  \n",input_i,finput_i,l_input_i);}

	float l_input_thre=2;
	float rot = l_input_i*l_input_thre/OUTPUT_SIZE_IN_PIXELS*0.08;
	;//input_thre;
	//if (gid_x==1000 && gid_y==1000){printf("gid_x %d - gid_y %d - gDx %4.4f - gDy %4.4f - rot %4.4f \n",gid_x,gid_y,x,y,rot);}
	cfloat_t C;
	C.x=x;
	C.y=y;
	rgb m;
	cfloat_t j;
	j.x=-0.8+rot;
	j.y=-0.156-rot;
	//j.y=-0.8-rot*3;
	//j=cfloat_mul(j,cfloat_powr(j,rot));
	//m=julia_iterations(C,j,rot,input_i,input_thre);
	m=julia_iterations(C,j,rot,100,l_input_thre);
	if (gid_x==750 && gid_y==750 && gid_z==1 )
		{printf("iterations %d coeff %4.1f red %d green %d blue %d  \n",m.iters,m.coeff,m.r,m.g,m.b);}

	uint pos0 = gid_x+gid_y*OUTPUT_SIZE_IN_PIXELS;
	//uint pos = pos0 + OUTPUT_SIZE_IN_PIXELS*OUTPUT_SIZE_IN_PIXELS*gid_z;
	uint pos=pos0*RGB+gid_z;
	// uint pos0 = gid_x*RGB*gid_z;
	// uint pos = pos0 + OUTPUT_SIZE_IN_PIXELS*gid_y;
	if (gid_z==0){result[pos]=m.r;}
	if (gid_z==1){result[pos]=m.g;}
	if (gid_z==2){result[pos]=m.b;}
	//printf("gid_x |%d| - gid_y |%d| - gid_z |%d| - pos |%d| - pos0 |%d| m |%d| \n",gid_x,gid_y,gid_z,pos,pos0, m);
}



