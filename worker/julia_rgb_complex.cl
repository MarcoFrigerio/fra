
#define MAX_ITERATIONS 30
#define MANDELBROT_THRESHOLD 2
#define OUTPUT_SIZE_IN_PIXELS_X 2000
#define OUTPUT_SIZE_IN_PIXELS_Y 2000
#define POWR 2
#define SPEEDF 0.3
#define RGB 4
#define MANDELBROT 0

typedef double2 cl_double_complex;
typedef cl_double_complex cl_complex;

inline double cl_complex_real_part(const cl_complex* n){
	return n->x;
}

inline double cl_complex_imaginary_part(const cl_complex* n){
	return n->y;
}
inline cl_complex cl_complex_add(const cl_complex* a, const cl_complex* b){
	return (cl_complex)( a->x + b->x, a->y + b->y );
}
inline cl_complex cl_complex_multiply(const cl_complex* a, const cl_complex* b){
	return (cl_complex)(a->x*b->x - a->y*b->y,  a->x*b->y + a->y*b->x);
}
// inline cl_complex cl_complex_ipow(const cl_complex* base ,  int exp ){
// 	cl_complex res;
// 	res.x=res.y=1;
// 	while(exp){
// 		if(exp & 1)
// 			res=cl_complex_multiply(&res,base);
// 		exp>>=1;
// 		res = cl_complex_multiply(&res,&res);
// 		}
	
// 	return res;
// }

inline cl_complex cl_complex_ipow(const cl_complex* base ,  int exp ){
	cl_complex res;
	res.x=base->x;
	res.y=base->y;
	while(exp>1){
		res=cl_complex_multiply(&res,base);
		exp--;
		}
	
	return res;
}
/*
 * Returns modulus of complex number (its length):
 */
inline double cl_complex_modulus(const cl_complex* n){
	return (sqrt( (n->x*n->x)+(n->y*n->y) ));
}


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
	// cfloat_t C,
	cl_complex C,
	cl_complex j,
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
		float absz;
		const float maxrgb = 255.0;
		// cfloat_t one;
		// one.x=20/iterations;
		// one.y=0;
		// one=cfloat_add(j,one);
		// cfloat_t Z = C;
		cl_complex Z = C;
		#pragma unroll
		//while (iterations < MAX_ITERATIONS){
		while (iterations < input_i){
			Z=cl_complex_ipow(&Z,POWR);
			// Z=cl_complex_multiply(&Z,&Z);
			Z =cl_complex_add(&Z,&j);
			absz=cl_complex_modulus(&Z);
			if (iterations==input_i-1){speediter1=absz;}
			speed+=absz+speed0;
			speed0=absz;
			if(absz > input_thre){break;}
			iterations++;
			iterfloat=iterations;
			}
		
		float coeff=(iterfloat/input_i)*128;
		// float absz = cfloat_abs(Z);
		// float absz = cl_romplex_modulus(&Z);
		float r;float g; float b;
		if (coeff<128){
			r=fmin(maxrgb,coeff*absz);
			float Z2=Z.y*Z.y;
			g=fmin(maxrgb,50*Z2);
			b=255-fmin(absz*52,maxrgb);
			// rrgb.alpha=fmax(fmax(r,g),b);
			}
		else{
			r=255-fmin(maxrgb,255*speediter1);//native_cos(absz)*255;
			g=fmin(maxrgb,255*speediter1);
			b=fmin(maxrgb,speed);
			}
		rrgb.r=r;
		rrgb.g=g;
		rrgb.b=b;
		rrgb.alpha=255;
		rrgb.iters=iterations;
		rrgb.coeff=absz;
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
	
	double x = gDx[gid_x];
	double y = gDy[gid_y];
	// if (gid_x==750 && gid_y==750 && gid_z==1 )
	// 	{printf("Start process %d %d",gid_x,gid_y);}

	if (gid_x==1 && gid_y==1 && gid_z==1 && input_i==1000)
	{
		uint wkdim=get_work_dim ();
		size_t ls0=get_local_id(0);
		size_t ls1=get_local_id(1);
		size_t ls2=get_local_id(2);

		printf("wkdim %d \n",wkdim);
		printf("par 0 %d \n",ls0);
		printf("par 1 %d \n",ls1);
		printf("par 2 %d \n",ls2);
		uint sx=sizeof(x);
		printf("size of x %d \n",sx);
	}


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
	cl_complex C;
	// cfloat_t C;
	C.x=y;
	C.y=x;
	__local rgb m;
	// cfloat_t j;
	cl_complex j;
	// j.x=-0.8+rot;
	// j.y=-0.156-rot;	
	j.x=-0.835+rot;
	j.y=-0.232-rot;	
	if (MANDELBROT==1) {j=C;}
	//j=cfloat_mul(j,cfloat_powr(j,rot));
	//m=julia_iterations(C,j,rot,input_i,input_thre);
	if (lid_z==0) {m=julia_iterations(C,j,rot,MAX_ITERATIONS,l_input_thre);}
	// if (gid_x==OUTPUT_SIZE_IN_PIXELS_X/2 && gid_y==OUTPUT_SIZE_IN_PIXELS_Y/2 && gid_z==1 )
	// 	{printf("x %lf y %lf iterations %d input_i %d red %d green %d blue %d coeff %4.4f j.x %1.4f j.y %1.4f \n",
	// 	x,y,m.iters,input_i,m.r,m.g,m.b,m.coeff,j.x,j.y);}

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



