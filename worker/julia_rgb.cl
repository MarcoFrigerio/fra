
#include <pyopencl-complex.h>
#define MAX_ITERATIONS 30
#define MANDELBROT_THRESHOLD 2
#define OUTPUT_SIZE_IN_PIXELS_X 2000
#define OUTPUT_SIZE_IN_PIXELS_Y 2000
#define POWR 2
#define SPEEDF 0.3
#define RGB 4
#define MANDELBROT 0

typedef struct {
	uint r;
	uint g;
	uint b;
	uint alpha;
	uint iters;
	float coeff;
}
rgb;

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
		float speediter1;
		const float maxrgb = 255.0;
		// cfloat_t one;
		// one.x=20/iterations;
		// one.y=0;
		// one=cfloat_add(j,one);
		cfloat_t Z = C;
		#pragma unroll
		//while (iterations < MAX_ITERATIONS){
		while (iterations < input_i){
			//Z = cfloat_divide(cfloat_mul(Z,Z),cfloat_cos(Z));
			Z=cfloat_powr(Z,POWR);
			//Z= cfloat_mul(j ,cfloat_exp(Z));
			//Z=cfloat_tan(Z);
			//Z = cfloat_mul(Z,Z);
			//Z = cfloat_mul(Z,Z);
			//Z = cfloat_cos(Z,cfloat_e(j));
			if (MANDELBROT==0){Z = cfloat_add(Z,j);}
			else if (MANDELBROT==1) {Z = cfloat_add(Z,C);}
			if (iterations==input_i-1){speediter1=cfloat_abs(Z);}
			speed+=cfloat_abs(Z)+speed0;
			speed0=cfloat_abs(Z);
			//if(iterations==1){speed=cfloat_abs(Z);}
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
			r=fmin(maxrgb,coeff*absz*2);
			g=fmin(maxrgb,25*Z.y*Z.y);
			b=255-fmin(absz*52,maxrgb);
			// rrgb.alpha=fmax(fmax(r,g),b);
			}
		else{
			// r=fmax(absz*32,255);
			// g=fmax(255-128*absz,255);
			// b=fmax(absz*48,255);
			r=255-fmin(maxrgb,255*speediter1);//native_cos(absz)*255;
			g=fmin(maxrgb,255*speediter1);
			b=fmin(maxrgb,speed);
			// rrgb.alpha=fmax(fmax(r,g),b);
			}
		rrgb.r=r;
		rrgb.g=g;
		rrgb.b=b;
		rrgb.alpha=255;
		rrgb.iters=iterations;
		rrgb.coeff=rrgb.alpha;
		// if (coeff<128){rrgb.r=coeff*absz;rrgb.g=absz*50;rrgb.b=(2-absz)*128;}
		// else{rrgb.r=absz*256;rrgb.g=absz*128;rrgb.b=(2-absz)*128;}
		// // // rrgb.r=coeff;
		// rrgb.g=absz*50;
		// rrgb.b=(2-absz)*128;
		return rrgb;
	}

__kernel void julia(
	__global double * gDx,
	__global double * gDy,
	const unsigned int input_i,
	const float input_thre,
	__global uint * result
	)
{

	uint gid_x = get_global_id(0);
	uint gid_y = get_global_id(1);
	uint gid_z = get_global_id(2);
	uint lid_z = get_local_id(2);	
	
	// if (gid_x==750 && gid_y==750 && gid_z==1 )
	// 	{printf("Start process %d %d",gid_x,gid_y);}

	if (gid_x==1 && gid_y==1 && gid_z==1 && input_i==1)
	{
		uint wkdim=get_work_dim ();
		size_t ls0=get_local_id(0);
		size_t ls1=get_local_id(1);
		size_t ls2=get_local_id(2);

		printf("wkdim %d \n",wkdim);
		printf("par 0 %d \n",ls0);
		printf("par 1 %d \n",ls1);
		printf("par 2 %d \n",ls2);
	}


	double x = gDx[gid_x];
	double y = gDy[gid_y];
	float finput_i = input_i;
	finput_i=cos(finput_i) ;
	uint l_input_i=input_i;
	//uint l_input_i=2*finput_i+input_i;
	// if (gid_x==1000 && gid_y==1000 && gid_z==1 )
	// 	{printf("input_i %d finput_i  %4.4f l_input_i %d  \n",input_i,finput_i,l_input_i);}


	double l_input_thre=MANDELBROT_THRESHOLD;
	double rot = l_input_i*l_input_thre/OUTPUT_SIZE_IN_PIXELS_Y*SPEEDF;
	//float rot = l_input_thre/OUTPUT_SIZE_IN_PIXELS;
	;//input_thre;
	//if (gid_x==1000 && gid_y==1000){printf("gid_x %d - gid_y %d - gDx %4.4f - gDy %4.4f - rot %4.4f \n",gid_x,gid_y,x,y,rot);}
	cfloat_t C;
	C.x=y;
	C.y=x;
	__local rgb m;
	cfloat_t j;
	// j.x=-0.8+rot;
	// j.y=-0.156-rot;	
	j.x=-0.835+rot;
	j.y=-0.232-rot;	
	//j=cfloat_mul(j,cfloat_powr(j,rot));
	//m=julia_iterations(C,j,rot,input_i,input_thre);
	if (lid_z==0) {m=julia_iterations(C,j,rot,MAX_ITERATIONS,l_input_thre);}
	if (gid_x==OUTPUT_SIZE_IN_PIXELS_X/2 && gid_y==OUTPUT_SIZE_IN_PIXELS_Y/2 && gid_z==1 )
		{printf("x %4.4f y %4.4f iterations %d input_i %d red %d green %d blue %d coeff %4.4f j.x %1.4f j.y %1.4f \n",
		x,y,m.iters,input_i,m.r,m.g,m.b,m.coeff,j.x,j.y);}

	uint pos0 = gid_y+gid_x*OUTPUT_SIZE_IN_PIXELS_Y;
	//uint pos = pos0 + OUTPUT_SIZE_IN_PIXELS*OUTPUT_SIZE_IN_PIXELS*gid_z;
	uint pos=pos0*RGB+gid_z;
	// uint pos0 = gid_x*RGB*gid_z;
	// uint pos = pos0 + OUTPUT_SIZE_IN_PIXELS*gid_y;
	if (lid_z==0){result[pos]=m.r;}
	if (lid_z==1){result[pos]=m.g;}
	if (lid_z==2){result[pos]=m.b;}
	if (lid_z==3){result[pos]=m.alpha;}

	//printf("gid_x |%d| - gid_y |%d| - gid_z |%d| - pos |%d| - pos0 |%d| m |%d| \n",gid_x,gid_y,gid_z,pos,pos0, m);
}



