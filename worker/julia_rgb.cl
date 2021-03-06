
// void printstrhex(uchar * strin,uint l,uint p){
//     //while (key_g[i]!='\0') {
//     for (uint i=0;i<l;i++){
//         if(i==0){printf("\n");printf(" p %d :",p);}
//         printf("%X", strin[i]);
//         if(i==l){printf(" %d \n",i);}
//         }
// }

// typedef struct {float x; float y;} nC;
// //// just some try
// float absmio (float i){
//         if (i<0)
//             return (i=i*(-1));
//         else
//             return i;
//     }

// nC somma (nC a, nC b){
//         nC n;
//         n.x=a.x+b.x;
//         n.y=a.y+b.y;
//         return n;
//     }

// nC moltip (nC a, nC b){
//     nC c;
//     c.x=a.x*b.x-a.y*b.y;
//     c.y=a.x*b.y+b.x*a.y;
//     return c;
//     }
/// real main from here
#include <pyopencl-complex.h>
#define MAX_ITERATIONS 30
#define MANDELBROT_THRESHOLD 2
#define OUTPUT_SIZE_IN_PIXELS 2000
#define RGB 3

typedef struct {
  uint r;
  uint g;
  uint b;
} rgb;

rgb julia_iterations(cfloat_t C,cfloat_t j ){
	uint iterations = 1;
	rgb rrgb;
	// cfloat_t one;
	// one.x=20/iterations;
	// one.y=0;
	// one=cfloat_add(j,one);

	cfloat_t Z = C;
	#pragma unroll
	while (iterations < MAX_ITERATIONS){
		//Z = cfloat_divide(cfloat_mul(Z,Z),cfloat_cos(Z));
		//Z=cfloat_powr(Z,3);
		//Z=cfloat_tan(Z);
		Z = cfloat_mul(Z,Z);
		Z = cfloat_add(Z,j);
		//Z = cfloat_divide(Z,cfloat_tan(j));
		if(cfloat_abs(Z) > MANDELBROT_THRESHOLD){break;}
		iterations++;
		}
	float iterfloat=iterations;
	float coeff=(iterfloat/MAX_ITERATIONS)*128;
	float absz = cfloat_abs(Z);
	// if (coeff<128){rrgb.r=coeff*absz;rrgb.g=absz*100;rrgb.b=(2-absz)*128;}
	// else{rrgb.r=absz*256;rrgb.g=absz*128;rrgb.b=255;}
	if (coeff<128){rrgb.r=coeff*absz;rrgb.g=absz*Z.y;rrgb.b=(2-absz)*128;}
	else{rrgb.r=Z.x;rrgb.g=absz*128;rrgb.b=Z.y;}
	// rrgb.r=coeff;
	// rrgb.g=absz*50;
	// rrgb.b=(2-absz)*128;
	

	return rrgb;
	}

__kernel void mandelbrot(
	__global float * gD,
	__global uint * result
	)
{
	uint gid_x = get_global_id(0);
	uint gid_y = get_global_id(1);
	uint gid_z = get_global_id(2);


	float x = gD[gid_x];
	float y = gD[gid_y];

	//printf("gid_x %d - gid_y %d - gDx %4.4f - gDy %4.4f \n",gid_x,gid_y,x,y);
	cfloat_t C ;    
	C.x=x;
	C.y=y;
	rgb m;
	cfloat_t j;
	j.x=-0.8;
	j.y=-0.156;
	m=julia_iterations(C,j);
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



