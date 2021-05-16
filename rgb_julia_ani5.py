#!/usr/bin/python3
import json
import math
from distutils import util
import multiprocessing
from types import FrameType
import PIL
# from numpy.core.fromnumeric import repeat
import pyopencl as cl
import numpy as np
import os
import matplotlib.pyplot as plt
# from PIL import Image
import matplotlib.animation as animation
from multiprocessing import Pool
from multiprocessing import Process
# from multiprocessing import Queue
from multiprocessing import set_start_method
# from multiprocessing import get_context
from PIL import Image
import time
import datetime as dt
from datetime import datetime
import math
import concatenate as coca
from colorama import Fore
import shutil
# import julia_parm as s

class opencl_py:
	PYOPENCL_COMPILER_OUTPUT='1' # set to '1' to see the openCL compile errors
	os.environ['PYOPENCL_COMPILER_OUTPUT'] = PYOPENCL_COMPILER_OUTPUT


	def __init__(self,platform,func,OUTPUT_SIZE_IN_PIXELS_X,OUTPUT_SIZE_IN_PIXELS_Y,CX,CY,RGB,FLAG_ROTATE,SPEEDF):
		self.OUTPUT_SIZE_IN_PIXELS_X=OUTPUT_SIZE_IN_PIXELS_X
		self.OUTPUT_SIZE_IN_PIXELS_Y=OUTPUT_SIZE_IN_PIXELS_Y
		self.CX=CX
		self.CY=CY
		self.RGB=RGB
		self.FLAG_ROTATE=FLAG_ROTATE
		self.SPEEDF=SPEEDF

		platforms = cl.get_platforms()
		if (platform > len(platforms)):
			assert("Selected platform %d doesn't exist" % platform)


		devices = platforms[platform].get_devices()
		# Create context for GPU/CPU
		#print("Using Platform %d:" % platform)
		self.ctx = cl.Context(devices)


		# Create queue for each kernel execution, here we only use 1 device
		self.queue = cl.CommandQueue(self.ctx,devices[0],cl.command_queue_properties.PROFILING_ENABLE)
		if func=="julia":filen="julia_rgb.cl"
		if func=="julia_c":filen="julia_rgb_complex.cl"

		self.OPENCL_CODE_PATH=os.path.join("worker",filen)

	def compile(self,marcos=dict,writeProcessedOpenCLCode=True):
		ori_src =""
		with open(self.OPENCL_CODE_PATH, "r") as rf:
			ori_src += rf.read()

		proc_src=""
		for line in ori_src.splitlines():
			if marcos:# processed all the needed marcos
				for k,v in marcos.items():
					if line.startswith("#define "+k+" "):
						line="#define "+k+" "+v# re-define marcos
						del(marcos[k])
						break
			proc_src += line+"\n"
		if marcos:
			print("Error! No matched marcos in "+self.OPENCL_CODE_PATH+" :")
			for k,v in marcos.iteritems():
				print(k)
		if writeProcessedOpenCLCode:
			with open(os.path.join(os.path.dirname(self.OPENCL_CODE_PATH),"processed.cl"), "w", encoding='utf-8') as f:
				f.write(proc_src)
		print("COMPILING KERNEL..")
		# Kernel function instantiation
		self.prg = cl.Program(self.ctx, proc_src).build()
		print("KERNEL COMPILED")



	def run_julia(self,input_i,thre,x_range,y_range,jiter):
		def fjx(input_i):
			if self.FLAG_ROTATE:return np.float64(math.pow(math.cos(input_i),2)*math.sin(input_i)*self.SPEEDF)
			return np.float64(0)

		def fjy(input_i):
			if self.FLAG_ROTATE:return np.float64(math.pow(math.sin(input_i),2)*self.SPEEDF)
			return np.float64(0)

		julia_shape=(self.OUTPUT_SIZE_IN_PIXELS_X,self.OUTPUT_SIZE_IN_PIXELS_Y,self.RGB)
		if self.RGB==3:workgroup_shape=(16,16,1)
		if self.RGB==4:workgroup_shape=(8,16,4)
		mf = cl.mem_flags# opencl memflag enum
		# matrix_generation_domain = np.linspace(-MANDELBROT_THRESHOLD, MANDELBRT_THRESHOLD, num=OUTPUT_SIZE_IN_PIXELS)
		# zoom=1-(c-1)/c

		matrix_generation_domain_x = np.linspace(-x_range+self.CX, x_range+self.CX, num=self.OUTPUT_SIZE_IN_PIXELS_X,dtype=np.float64)
		matrix_generation_domain_y = np.linspace(-y_range+self.CY, y_range+self.CY, num=self.OUTPUT_SIZE_IN_PIXELS_Y,dtype=np.float64)

		# matrix_generation_domain_x=matrix_generation_domain_x
		# matrix_generation_domain_x=matrix_generation_domain_y
		gD_npx = np.array(matrix_generation_domain_x,dtype=np.float64)
		gD_gx = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=gD_npx)

		gD_npy = np.array(matrix_generation_domain_y,dtype=np.float64)
		gD_gy = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=gD_npy)

		input_ib=np.float64(input_i)
		# input_thre=np.float32(thre)
		input_jiter=np.float32(jiter)

		rotx_i=fjx(input_i)
		roty_i=fjy(input_i)

		result = np.empty(julia_shape, dtype=np.uint32)	
		result_g = cl.Buffer(self.ctx, mf.WRITE_ONLY,result.nbytes)# size should be in byte
		
		start_event=cl.enqueue_marker(self.queue)

		finish_event=self.prg.julia(self.queue,
			julia_shape,
			# (1,1,4), 
			workgroup_shape,
			# None,
			gD_gx,
			gD_gy,
			input_ib,
			input_jiter,
			rotx_i,
			roty_i,
			result_g )
		finish_event.wait()
		
		rt = cl.enqueue_copy(self.queue, result, result_g)
		gD_gx.release()
		gD_gy.release()
		result_g.release()
		return result

def run_julia_py(input_i,x_range,y_range,jiter,OUTPUT_SIZE_IN_PIXELS_X,OUTPUT_SIZE_IN_PIXELS_Y,RGB,SPEEDF,CX,CY,FLAG_ROTATE):
	def fjx(input_i):
		if FLAG_ROTATE:return np.float64(math.pow(math.cos(input_i),2)*math.sin(input_i)*SPEEDF)
		return np.float64(0)

	def fjy(input_i):
		if FLAG_ROTATE:return np.float64(math.pow(math.sin(input_i),2)*SPEEDF)
		return np.float64(0)

	matrix_generation_domain_x = np.linspace(-x_range+CX, x_range+CX, num=OUTPUT_SIZE_IN_PIXELS_X,dtype=np.float64)
	matrix_generation_domain_y = np.linspace(-y_range+CY, y_range+CY, num=OUTPUT_SIZE_IN_PIXELS_Y,dtype=np.float64)
	rotx_i=fjx(input_i)
	roty_i=fjy(input_i)

	gD_npx = np.array(matrix_generation_domain_x,dtype=np.float64)
	gD_npy = np.array(matrix_generation_domain_y,dtype=np.float64)
	input_ib=np.float64(input_i)
	input_jiter=np.float32(jiter)
	julia_shape=(OUTPUT_SIZE_IN_PIXELS_X,OUTPUT_SIZE_IN_PIXELS_Y,RGB)
		
	result = np.empty(julia_shape, dtype=np.uint32)	
	ix=0
	for x in matrix_generation_domain_x:
		iy=0
		for y in matrix_generation_domain_y:
			iters=0
			z=complex(0,0)
			c=complex(y,x)
			while iters < 100:
				z=z**2+c
				if abs(z)>2:break
				iters+=1
			
			perciters=(iters/100)*255
			result[ix,iy,0]=int(perciters)
			result[ix,iy,1]=0			
			result[ix,iy,2]=0
			iy+=1
		ix+=1
	return result


def save_file( dir,filename,result_matrix,fig,ims,ccycle,figuresize_x,figuresize_y):
	print("LOOP ANIMATION LOAD....")
	nr_im=len(result_matrix)
	ims=[]
	for i in range(nr_im):
		# im[i].write_png("img/test"+str(i)+".png")
		# result_matrix=rescale_linear(result_matrix)
		# print(f"processing matrix {i}")
		# print(result_matrix[i].shape)
		# img.show()
		# print(f"image {i}")
		# iml.append(img)
		if i==nr_im-1:
			# if julia.RGB==3:rgb='RGB'
			# if julia.RGB==4:rgb='RGBA'
			img=Image.fromarray(result_matrix[i].astype('uint8'))
			# img.show()
			img.save(dir+str(i)+".png")
		im=plt.imshow(result_matrix[i],animated=True,interpolation="bilinear")
		# plt.show()
		ims.append([im])
	del result_matrix
	print("END LOOP...SAVING FILE....")
	try:
		plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
	except:
		pass	
	plt.axis("off")
	ani = animation.ArtistAnimation(fig, ims, interval=0, blit=True,repeat_delay=0,repeat=True)
	ani.save(dir+filename,fps=60,extra_args=["-threads", "4"])
	# ani.save(dir+filename,fps=60)

def actual_time(start):
	return datetime.now()-start

def printtime(actual_time):
	# return dt.timedelta(actual_time)
	return str(actual_time).split('.', 2)[0]



def julia():

	def calc_xrange(input_i):
		return (MAX-input_i)/(MAX+input_i*100)	

	def calc_zoom(xrange,z):
		return np.float64((xrange-z)/(100*z+xrange))

	f="julia_parm.json"
	with open(f) as f:
		params=json.load(f)
	RGB=int(params["RGB"])
	OUTPUT_SIZE_IN_PIXELS_X = int(params["OUTPUT_SIZE_IN_PIXELS_X"])
	OUTPUT_SIZE_IN_PIXELS_Y = int(params["OUTPUT_SIZE_IN_PIXELS_Y"])  # 2k number of rows
	X_RANGE=float(params["X_RANGE"])                   # initial start range of y values 
	MAX_ITERATIONS =int(params["MAX_ITERATIONS"])             # If 0 then it s dinamic. Else, it s max number of iterations in single pixel opencl calculation. 
	MINJITER=int(params["MINJITER"])               # if dinamic, initial number of iterations 
	MAXJITER=int(params["MAXJITER"])             # if dinamic, final number of iterations 
	MANDELBROT_THRESHOLD = int(params["MANDELBROT_THRESHOLD"])        # thresold of the absolute value of reiterated Z=
	MIN=int(params["MIN"])                       # start point of C values 
	MAX=int(params["MAX"])                        # end point of C values
	FRAMEEVERY=int(params["FRAMEEVERY"])                   # number of frames not calculated between two calculated
	CYCLEFRAMEBASE=int(params["CYCLEFRAMEBASE"])
	CYCLEFRAME=CYCLEFRAMEBASE*FRAMEEVERY
	SPEEDF =float(params["SPEEDF"])                    # max delta of change of C value in julia set	
	POWR=int(params["POWR"])                          # powr of Z in iteration function
	CX=float(params["CX"])                          # position of x center (good for julia set)
	CY=float(params["CY"])                          # position of x center (good for julia set)
	DIR=params["DIR"]                   # working dir
	MANDELBROT= int(params["MANDELBROT"])                     # 1 = mandelbrot set , 0 = julia set
	FLAG_ZOOM=bool(util.strtobool(params["FLAG_ZOOM"]))                  # Flag Zoom the image
	FLAG_ROTATE=bool(util.strtobool(params["FLAG_ROTATE"]))               #apply a movement to j values
	JX=float(params["JX"])
	JY=float(params["JY"])
	EXPZOOMSTART=float(params["EXPZOOMSTART"])
	EXPZOOM=float(params["EXPZOOM"])


	# set_start_method("spawn")
	try:shutil.rmtree(DIR)
	except:pass
	os.mkdir(DIR)
	assert (RGB==3 or RGB==4)

	loops=MAX-MIN
	opencl_ctx=opencl_py(0,'julia_c',OUTPUT_SIZE_IN_PIXELS_X,OUTPUT_SIZE_IN_PIXELS_Y,CX,CY,RGB,FLAG_ROTATE,SPEEDF)
	opencl_ctx.compile({"OUTPUT_SIZE_IN_PIXELS_X":str(OUTPUT_SIZE_IN_PIXELS_X),
						"OUTPUT_SIZE_IN_PIXELS_Y":str(OUTPUT_SIZE_IN_PIXELS_Y),
						"MAX_ITERATIONS":str(MAX_ITERATIONS),
						"MANDELBROT_THRESHOLD":str(MANDELBROT_THRESHOLD),
						"SPEEDF":str(SPEEDF),
						"MANDELBROT":str(MANDELBROT),
						"POWR":str(POWR),
						"RGB":str(RGB),
						"JX":str(JX),
						"JY":str(JY)
						})

	figuresize_y=OUTPUT_SIZE_IN_PIXELS_X/100
	figuresize_x=OUTPUT_SIZE_IN_PIXELS_Y/100
	screen_format=OUTPUT_SIZE_IN_PIXELS_Y/OUTPUT_SIZE_IN_PIXELS_X

	if loops>CYCLEFRAME:
		cycleframe=CYCLEFRAME
		frameevery=FRAMEEVERY
	else:
		cycleframe=CYCLEFRAMEBASE
		frameevery=1

	start=datetime.now()
	nrloops=loops//cycleframe		
	counter=0
	ccycle=0
	countertot=0
	video_list=[]
	jobs=[]
	cloops=loops//frameevery
	expl=np.linspace(EXPZOOMSTART,EXPZOOM, num=cloops,dtype=np.float64)
	jiterl=np.linspace(MINJITER,MAXJITER, num=cloops,dtype=np.float64)
	rotlnsp=np.linspace(0,math.pi*2, num=cloops,dtype=np.float64)
	for _ in range(nrloops):
		min=MIN+ccycle*cycleframe
		max=min+cycleframe
		result_matrix=[]
		cor=1
		for i in range (min,max,frameevery):
			if FLAG_ZOOM:
				xrange=calc_xrange(i)
				zoomnp=np.linspace(0,xrange, num=cloops,dtype=np.float64)
				z=np.float64(zoomnp[counter])
				zoom=calc_zoom(xrange,z)
				# print(f"i {zoom}"):
				exp=np.float64(expl[counter])
				zoom=np.float64(zoom**exp)
				x_range=np.float64(xrange*(zoom))
				y_range=np.float64(x_range*screen_format)
				jiter=jiterl[counter]
			else:
				xrange=x_range=X_RANGE
				y_range=x_range*screen_format
				z=0
				zoom=0
				jiter=0
			perc=i/MAX
			estimated_time=actual_time(start)*(MAX/i) - actual_time(start)
			print(f"{Fore.YELLOW}{perc:.0%} {i:,}/{MAX:,} {Fore.CYAN} {cor}/{CYCLEFRAMEBASE} {Fore.RESET} {Fore.GREEN}{printtime(actual_time(start))}{Fore.RESET} {Fore.RED}{printtime(estimated_time)} {Fore.RESET} \
init xrange {xrange} desc zoom : {zoom} - new xrange {x_range}")
			# input_i = counter/cloops
			# input_i = counter
			input_i=rotlnsp[counter]
			result_matrix.append(opencl_ctx.run_julia(input_i,i/50,x_range,y_range,jiter))
			# result_matrix.append(run_julia_py(input_i,x_range,y_range,jiter,OUTPUT_SIZE_IN_PIXELS_X,OUTPUT_SIZE_IN_PIXELS_Y,RGB,SPEEDF,CX,CY,FLAG_ROTATE))

			cor+=1
			counter+=1

		# with Pool() as pj:
		# 	result = pj.map(run_julia_py(input_i,x_range,y_range,jiter,OUTPUT_SIZE_IN_PIXELS_X,OUTPUT_SIZE_IN_PIXELS_Y,RGB,SPEEDF,CX,CY,FLAG_ROTATE),[])
		# 	result=np.array(result)
		# 	result_matrix.append(result)
		ims = []
		fig=plt.figure(figsize=(figuresize_x, figuresize_y))
		fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
		sstcy=str(ccycle).rjust(5,'0')
		filen='julia'+sstcy+'.mp4'
		video_list.append(filen)
		filename='julia'+sstcy+'.mp4'

		while True:
			pcs = len(multiprocessing.active_children())
			if pcs<4:
				p = Process(target=save_file,args=(DIR,filename,result_matrix,fig,ims,ccycle,figuresize_x,figuresize_y,))
				jobs.append(p)
				p.start()
				break
			time.sleep(1)

		ccycle+=1
	print("WAITING FOR ALL JOBS TO FINISH...")
	for job in jobs:job.join()
	print("CREATING VIDEO...")
	mean_time_for_frame=actual_time(start)/counter
	out=coca.concatenate(DIR,video_list)
	if out==0:print(f"{Fore.LIGHTGREEN_EX}VIDEO CREATED! {Fore.RESET} ")
	else:print("{Fore.RED}ERROR IN VIDEO CREATION!!! {Fore.RESET} ")
	print(f"Elapsed {Fore.GREEN}{printtime(actual_time(start))} -  Mean time for frame {printtime(mean_time_for_frame)}{Fore.RESET}")

if __name__ == "__main__":
	julia()