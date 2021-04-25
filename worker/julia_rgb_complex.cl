
#define MAX_ITERATIONS 30
#define MANDELBROT_THRESHOLD 2
#define OUTPUT_SIZE_IN_PIXELS_X 2000
#define OUTPUT_SIZE_IN_PIXELS_Y 2000
#define POWR 2
#define SPEEDF 0.3
#define RGB 4
#define MANDELBROT 0
#define JX 0
#define JY 0
#define PI 1.570796326794896619231321691639751442098584699

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
		float mean_speed;
		const float maxrgb = 255.0;
		// cfloat_t one;
		// one.x=20/iterations;
		// one.y=0;
		// one=cfloat_add(j,one);
		// cfloat_t Z = C;
		cl_complex Z;
		if  (MANDELBROT==1){Z.x=0;Z.y=0;}
		else {Z=C;}
		#pragma unroll
		// while (iterations < MAX_ITERATIONS){
		while (iterations < input_i){
			Z=cl_complex_ipow(&Z,POWR);
			// Z=cl_complex_multiply(&Z,&Z);
			// Z =cl_complex_add(&Z,&C);
			Z =cl_complex_add(&Z,&j);
			absz=cl_complex_modulus(&Z);
			if (iterations==input_i-1){speediter1=absz;}
			speed+=absz+speed0;
			speed0=absz;
			mean_speed=speed/iterations;
			if(absz > input_thre){break;}
			iterations++;
			iterfloat=iterations;
			}
		
		float perit = iterfloat/input_i;
		float coeff=perit*255;
		float i_x=perit;
		float pi_x=perit*PI;
		float r;float g; float b;
		float xr=((-1)*pown((i_x-1),4)+1);
		float yr=5/(exp(-1*pi_x+2))-5/3;
		float par01 = 2.7;
		// float xr=maxrgb*i_x;
		r=fmin(maxrgb,maxrgb*xr);
		g=fmax(fmin(maxrgb-75*absz*cos(pi_x),maxrgb),0);
		// g=fmin(maxrgb,maxrgb*sin(pi_x)/(xr*absz));
		// g=fmin(maxrgb,maxrgb*mean_speed*powr(cos(pi_x),12));
		// g=fmin(255-4*absz*log2(speed*mean_speed),maxrgb);
		// b=fmin(maxrgb,maxrgb*par01*mean_speed*sin(pi_x));			
		b=fmin(maxrgb,maxrgb*par01*mean_speed*yr);			
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
	const double input_i,
	const float input_jiter,
	const double rotx,
	const double roty,
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

	if (gid_x==1 && gid_y==1 && gid_z==1 && input_i==1)
	{
		uint wksize=get_work_dim ();
		uint ls1=get_local_size(1);
		uint ls0=get_local_size(0);
		uint ls2=get_local_size(2);

		printf("wksize %d \n",wksize);
		printf("par 0 %d \n",ls0);
		printf("par 1 %d \n",ls1);
		printf("par 2 %d \n",ls2);
		uint sx=sizeof(x);
		printf("size of x %d \n",sx);
	}

	// double rotx = pown(cos(input_i),3)*sin(input_i)*SPEEDF;
	// double rotx = cos(input_i)*sin(input_i)*SPEEDF;
	// double roty =sin(input_i)*SPEEDF;

	cl_complex C;
	// cfloat_t C;
	C.x=y;
	C.y=x;
	cl_complex j;
	// j.x=-0.835+rot;
	// // j.y=-0.232-rot;	
	// j.x=0+rot;
	// j.y=0-rot;
	// j.x=-1.76938317919551501821384728608547378290574726365475143746552821652789971538042486160358350056705;
	// j.y=0.00423684791873677221492650717136799707668267091740375727945943565011165050579686460572594185089;
	j.x=JX-rotx;
	j.y=JY+roty;
	if (MANDELBROT==1) {j=C;}
	// else{j=cl_complex_add(&C,&j);}
	// __local rgb m;
	// if (lid_z==0) {m=julia_iterations(C,j,rot,MAX_ITERATIONS,l_input_thre);}
	rgb m;
	// m=julia_iterations(C,j,MAX_ITERATIONS,MANDELBROT_THRESHOLD);
	uint uint_jiters = (uint) input_jiter;
	m=julia_iterations(C,j,input_jiter,MANDELBROT_THRESHOLD);

	if (gid_x==OUTPUT_SIZE_IN_PIXELS_X/2 && gid_y==OUTPUT_SIZE_IN_PIXELS_Y/2 && gid_z==1 )
		{printf("x %f y %f iterations %d/%d input_i %.8f red %d green %d blue %d par_par %4.4f j.x %f j.y %f \n",
		x,y,m.iters,uint_jiters,input_i,m.r,m.g,m.b,m.coeff,j.x,j.y);}
	if (gid_x==0 && gid_y==0 && gid_z==1 )
		{printf("x %f y %f iterations %d/%d input_i %.8f red %d green %d blue %d par_par %4.4f j.x %5.4f j.y %5.4f \n",
		x,y,m.iters,uint_jiters,input_i,m.r,m.g,m.b,m.coeff,j.x,j.y);}
	uint pos0 = gid_y+gid_x*OUTPUT_SIZE_IN_PIXELS_Y;
	uint pos=pos0*RGB+gid_z;

	// if (lid_z==0){result[pos]=m.r;}
	// if (lid_z==1){result[pos]=m.g;}
	// if (lid_z==2){result[pos]=m.b;}
	// if (lid_z==3){result[pos]=m.alpha;}
	if (gid_z==0){result[pos]=m.r;}
	if (gid_z==1){result[pos]=m.g;}
	if (gid_z==2){result[pos]=m.b;}
	if (gid_z==3 && RGB==4){result[pos]=m.alpha;}


	//printf("gid_x |%d| - gid_y |%d| - gid_z |%d| - pos |%d| - pos0 |%d| m |%d| \n",gid_x,gid_y,gid_z,pos,pos0, m);
}



