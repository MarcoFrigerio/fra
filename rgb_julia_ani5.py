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
import math
import concatenate as coca


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
		julia_shape=(OUTPUT_SIZE_IN_PIXELS_X,OUTPUT_SIZE_IN_PIXELS_Y,4)
		mf = cl.mem_flags# opencl memflag enum
		# matrix_generation_domain = np.linspace(-MANDELBROT_THRESHOLD, MANDELBROT_THRESHOLD, num=OUTPUT_SIZE_IN_PIXELS)
		# zoom=1-(c-1)/c

		matrix_generation_domain_x = np.linspace(-x_range+CX, x_range+CX, num=OUTPUT_SIZE_IN_PIXELS_X,dtype=np.float64)
		matrix_generation_domain_y = np.linspace(-y_range+CY, y_range+CY, num=OUTPUT_SIZE_IN_PIXELS_Y,dtype=np.float64)

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
			(1,1,4),#None ,
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

def save_file( filename,result_matrix,fig,ims,ccycle,figuresize_x,figuresize_y):
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
			img = Image.fromarray(result_matrix[i].astype('uint8'), 'RGBA')
			img.save("img/"+str(i)+".png")
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
	ani.save(filename,fps=60,extra_args=["-threads", "4"])

# def concatenate(video_list):
# 	elenco_file_temp = []
# 	for f in video_list:
# 		file = "img/temp" + str(video_list.index(f) ) + ".ts"
# 		os.system("ffmpeg -y -i " + f + " -c copy -bsf:v h264_mp4toannexb -f mpegts " + file)
# 		elenco_file_temp.append(file)
# 	# print(elenco_file_temp)
# 	stringa = "ffmpeg -y -i \"concat:"
# 	# for f in elenco_file_temp:
# 	# 	stringa += f
# 	# 	# if elenco_file_temp.index(f) != len(elenco_file_temp)-1:
# 		# 	stringa += "|"
# 		# else:
# 		# 	stringa += "\" -c copy  -bsf:a aac_adtstoasc output.mp4"
# 		# for f in elenco_file_temp:
# 	input_file_list="img/input_file_list"
# 	with open(input_file_list,"w") as f:
# 		for fts in elenco_file_temp:
# 			f.write("file '"+fts.lstrip("img/")+"'\n")
		
# 	stringa = "ffmpeg -y -f concat -safe 0 -i "+input_file_list+" -c copy output.mp4 "

# 	# print(stringa)
# 	os.system(stringa)

if __name__ == "__main__":
	# OUTPUT_SIZE_IN_PIXELS_X = 1080 # number of columns
	# OUTPUT_SIZE_IN_PIXELS_Y = 1920 # number of rows
	OUTPUT_SIZE_IN_PIXELS_X = 1440  # number of columns
	OUTPUT_SIZE_IN_PIXELS_Y = 2560  # number of rows
	X_RANGE=1                   # initial start range of y values 
	MAX_ITERATIONS = 90             # max number of iterations in single pixel opencl calculation
	MANDELBROT_THRESHOLD = 2        # thresold of the absolute value of reiterated Z
	MIN=1                       # start point of C values 
	MAX=1_000_000_000                        # end point of C values
	SPEEDF = 0.1                    # speed of change of C value in julia set
	POWR=2                          # powr of Z in iteration function
	CX=0.01                          # position of x center (good for julia set)
	CY=-0.55                        # position of y center (good for julia set)
	CX=0.413238151606368892027      # position of y center (good for mandelbrot set)
	CY=-1.24254013716898265806      # position of y center	 (good for mandelbrot set)
	DIR="img/"                   # working dir
	CX=math.e/20
	CY=math.e/7
	# CX=0      # position of y center
	# CY=0      # position of y center	
	MANDELBROT=1                    # 1 = mandelbrot set , 0 = julia set
	FLAG_ZOOM=True                  # Flag Zoom the image
	FRAMEEVERY=5_000_000                   # number of frames not calculated between two calculated
	COMPLEX_CAL=True                 # calculation with custom complex opencl definition
	CYCLEFRAMEBASE=60
	CYCLEFRAME=CYCLEFRAMEBASE*FRAMEEVERY

	set_start_method("spawn")

	loops=MAX-MIN
	if COMPLEX_CAL:
		opencl_ctx=opencl_py(0,'julia_c')
	else:
		opencl_ctx=opencl_py(0,'julia')

	opencl_ctx.compile({"OUTPUT_SIZE_IN_PIXELS_X":str(OUTPUT_SIZE_IN_PIXELS_X),
						"OUTPUT_SIZE_IN_PIXELS_Y":str(OUTPUT_SIZE_IN_PIXELS_Y),
						"MAX_ITERATIONS":str(MAX_ITERATIONS),
						"MANDELBROT_THRESHOLD":str(MANDELBROT_THRESHOLD),
						"SPEEDF":str(SPEEDF),
						"MANDELBROT":str(MANDELBROT),
						"POWR":str(POWR)})

	figuresize_y=OUTPUT_SIZE_IN_PIXELS_X/100
	figuresize_x=OUTPUT_SIZE_IN_PIXELS_Y/100
	screen_format=OUTPUT_SIZE_IN_PIXELS_Y/OUTPUT_SIZE_IN_PIXELS_X


	if loops>CYCLEFRAME:
		cycleframe=CYCLEFRAME
		frameevery=FRAMEEVERY
	else:
		cycleframe=CYCLEFRAMEBASE
		frameevery=1

	start=time.time()
	nrloops=loops//cycleframe		
	counter=0
	ccycle=0
	video_list=[]
	jobs=[]
	for xcycle in range(nrloops):
		min=MIN+ccycle*cycleframe
		max=min+cycleframe
		result_matrix=[]
		for i in range (min,max,frameevery):
			if FLAG_ZOOM:
				xrange=np.float64((MAX-i)/(MAX+i*60))
				zoomnp=np.linspace(0,xrange, num=loops//frameevery,dtype=np.float64)
				z=np.float64(zoomnp[counter])
				zoom=np.float64((1-z)/(100*z+1))
				# print(f"i {zoom}"):
				exp=(10*i+MAX)/MAX
				zoom=zoom=np.float64(zoom**exp)
				x_range=np.float64(xrange*(zoom))
				y_range=np.float64(x_range*screen_format)
			else:
				xrange=X_RANGE
				z=0
			actual_time=round(time.time()-start)
			perc=i/MAX
			print(f"{i:,}/{MAX:,} {perc:.0%} seconds {actual_time} init xrange {xrange} desc zoom : {zoom} - new xrange {x_range}")
			result_matrix.append(opencl_ctx.run_julia(i,i/50,x_range,y_range))
			counter+=1
			#f_matrix_gen((opencl_ctx,i,i/40)))
		# print("RESCALING RGB VALUES..")
		# with Pool() as p:
		# 	result_matrix=p.map(rescal	e_linear,result_matrix)
		# print("BEGIN PLOTTING IMAGING..")

		# ani.save('julia.mp4',fps=30,extra_args=["-threads", "4"])
		# with Pool() as p:
		# 	p.map(save_file,[(ims,ccycle,figuresize_x,figuresize_y)])

		ims = []
		fig=plt.figure(figsize=(figuresize_x, figuresize_y))
		fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
		sstcy=str(ccycle).rjust(5,'0')
		filen='julia'+sstcy+'.mp4'
		video_list.append(filen)
		filename=DIR+'julia'+sstcy+'.mp4'

		while True:
			pcs = len(multiprocessing.active_children())
			if pcs<4:
				p = Process(target=save_file,args=(filename,result_matrix,fig,ims,ccycle,figuresize_x,figuresize_y,))
				jobs.append(p)
				p.start()
				break
			time.sleep(1)
		# p.join()
		# del result_matrix
		# ani.save('img/julia'+str(ccycle)+'.mp4',fps=60,extra_args=["-threads", "4"])
		# ani.save('julia'+str(ccycle)+'.mp4',fps=15,extra_args=["-threads", "4","-codec","hevc"])
		ccycle+=1
	print("WAITING FOR ALL JOBS TO FINISH...")
	for job in jobs:
		job.join()
	print("CREATING VIDEO...")
	coca.concatenate(DIR,video_list)
	print("VIDEO CREATED!")
