#!/usr/bin/python3


import multiprocessing
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
import datetime
import math
import concatenate as coca
from colorama import Fore
import julia_parm as s


class opencl_py:
	PYOPENCL_COMPILER_OUTPUT='1' # set to '1' to see the openCL compile errors
	os.environ['PYOPENCL_COMPILER_OUTPUT'] = PYOPENCL_COMPILER_OUTPUT


	def __init__(self,platform,func):

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


	def run_julia(self,input_i,thre,x_range,y_range):
		julia_shape=(s.OUTPUT_SIZE_IN_PIXELS_X,s.OUTPUT_SIZE_IN_PIXELS_Y,4)
		mf = cl.mem_flags# opencl memflag enum
		# matrix_generation_domain = np.linspace(-MANDELBROT_THRESHOLD, MANDELBROT_THRESHOLD, num=OUTPUT_SIZE_IN_PIXELS)
		# zoom=1-(c-1)/c

		matrix_generation_domain_x = np.linspace(-x_range+s.CX, x_range+s.CX, num=s.OUTPUT_SIZE_IN_PIXELS_X,dtype=np.float64)
		matrix_generation_domain_y = np.linspace(-y_range+s.CY, y_range+s.CY, num=s.OUTPUT_SIZE_IN_PIXELS_Y,dtype=np.float64)

		# matrix_generation_domain_x=matrix_generation_domain_x
		# matrix_generation_domain_x=matrix_generation_domain_y
		gD_npx = np.array(matrix_generation_domain_x,dtype=np.float64)
		gD_gx = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=gD_npx)

		gD_npy = np.array(matrix_generation_domain_y,dtype=np.float64)
		gD_gy = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=gD_npy)

		input_ib=np.int32(input_i)
		input_thre=np.float32(thre)

		result = np.empty(julia_shape, dtype=np.uint32)	
		result_g = cl.Buffer(self.ctx, mf.WRITE_ONLY,result.nbytes)# size should be in byte
		
		start_event=cl.enqueue_marker(self.queue)

		finish_event=self.prg.julia(self.queue,
			julia_shape,
			# (1,1,4), 
			(8,16,4), 
			# None,
			gD_gx,
			gD_gy,
			input_ib,
			input_thre,
			result_g )
		finish_event.wait()
		
		rt = cl.enqueue_copy(self.queue, result, result_g)
		gD_gx.release()
		gD_gy.release()
		result_g.release()
		return result

def printtime(actual_time):
	return datetime.timedelta(seconds=actual_time)

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
		if i==1:
			img=Image.fromarray(result_matrix[i].astype('uint8'), 'RGBA')
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

def actual_time(start):
	return round(round(time.time())-start)

if __name__ == "__main__":


	set_start_method("spawn")
	try:os.mkdir(s.DIR)
	except:pass

	loops=s.MAX-s.MIN
	if s.COMPLEX_CAL:
		opencl_ctx=opencl_py(0,'julia_c')
	else:
		opencl_ctx=opencl_py(0,'julia')

	opencl_ctx.compile({"OUTPUT_SIZE_IN_PIXELS_X":str(s.OUTPUT_SIZE_IN_PIXELS_X),
						"OUTPUT_SIZE_IN_PIXELS_Y":str(s.OUTPUT_SIZE_IN_PIXELS_Y),
						"MAX_ITERATIONS":str(s.MAX_ITERATIONS),
						"MANDELBROT_THRESHOLD":str(s.MANDELBROT_THRESHOLD),
						"SPEEDF":str(s.SPEEDF),
						"MANDELBROT":str(s.MANDELBROT),
						"POWR":str(s.POWR)})

	figuresize_y=s.OUTPUT_SIZE_IN_PIXELS_X/100
	figuresize_x=s.OUTPUT_SIZE_IN_PIXELS_Y/100
	screen_format=s.OUTPUT_SIZE_IN_PIXELS_Y/s.OUTPUT_SIZE_IN_PIXELS_X

	if loops>s.CYCLEFRAME:
		cycleframe=s.CYCLEFRAME
		frameevery=s.FRAMEEVERY
	else:
		cycleframe=s.CYCLEFRAMEBASE
		frameevery=1

	start=round(time.time())
	nrloops=loops//cycleframe		
	counter=0
	ccycle=0
	video_list=[]
	jobs=[]
	expl=np.linspace(1,3, num=loops//frameevery,dtype=np.float64)
	for xcycle in range(nrloops):
		min=s.MIN+ccycle*cycleframe
		max=min+cycleframe
		result_matrix=[]
		cor=1
		for i in range (min,max,frameevery):
			if s.FLAG_ZOOM:
				xrange=np.float64((s.MAX-i)/(s.MAX+i*20))
				zoomnp=np.linspace(0,xrange, num=loops//frameevery,dtype=np.float64)
				z=np.float64(zoomnp[counter])
				zoom=np.float64((xrange-z)/(50*z+xrange))
				# print(f"i {zoom}"):
				exp=np.float64(expl[counter])
				zoom=np.float64(zoom**exp)
				x_range=np.float64(xrange*(zoom))
				y_range=np.float64(x_range*screen_format)
			else:
				xrange=s.X_RANGE
				z=0
			perc=i/s.MAX
			estimated_time=round(actual_time(start)*(s.MAX/i) - actual_time(start))
			print(f"{Fore.YELLOW}{perc:.0%} {i:,}/{s.MAX:,} {Fore.CYAN} {cor}/{s.CYCLEFRAMEBASE} {Fore.RESET} {Fore.GREEN}{printtime(actual_time(start))}{Fore.RESET} {Fore.RED}{printtime(estimated_time)} {Fore.RESET} \
init xrange {xrange} desc zoom : {zoom} - new xrange {x_range}")
			result_matrix.append(opencl_ctx.run_julia(i,i/50,x_range,y_range))
			cor+=1
			counter+=1

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
				p = Process(target=save_file,args=(s.DIR,filename,result_matrix,fig,ims,ccycle,figuresize_x,figuresize_y,))
				jobs.append(p)
				p.start()
				break
			time.sleep(1)

		ccycle+=1
	print("WAITING FOR ALL JOBS TO FINISH...")
	for job in jobs:
		job.join()
	print("CREATING VIDEO...")
	mean_time_for_frame=actual_time(start)/counter
	out=coca.concatenate(s.DIR,video_list)
	if out==0:print(f"{Fore.LIGHTGREEN_EX}VIDEO CREATED! {Fore.RESET} ")
	else:print("{Fore.RED}ERROR IN VIDEO CREATION!!! {Fore.RESET} ")
	print(f"Elapsed {Fore.GREEN}{printtime(actual_time(start))} -  Mean time for frame {printtime(mean_time_for_frame)}{Fore.RESET}")
