#!/usr/bin/python3


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


	def run_julia(self,input_i,thre,c):
		julia_shape=(OUTPUT_SIZE_IN_PIXELS_X,OUTPUT_SIZE_IN_PIXELS_Y,4)
		mf = cl.mem_flags# opencl memflag enum
		# matrix_generation_domain = np.linspace(-MANDELBROT_THRESHOLD, MANDELBROT_THRESHOLD, num=OUTPUT_SIZE_IN_PIXELS)
		screen_format=OUTPUT_SIZE_IN_PIXELS_Y/OUTPUT_SIZE_IN_PIXELS_X
		# zoom=1-(c-1)/c
		zoom=(1-c)/(c+1)
		# print(f"i {zoom}")
		x_range=X_RANGE*(zoom)
		y_range=x_range*screen_format
		matrix_generation_domain_x = np.linspace(-x_range+CX, x_range+CX, num=OUTPUT_SIZE_IN_PIXELS_X)
		matrix_generation_domain_y = np.linspace(-y_range+CY, y_range+CY, num=OUTPUT_SIZE_IN_PIXELS_Y)

		# matrix_generation_domain_x=matrix_generation_domain_x
		# matrix_generation_domain_x=matrix_generation_domain_y
		gD_npx = np.array(matrix_generation_domain_x,dtype=np.float32)
		gD_gx = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=gD_npx)

		gD_npy = np.array(matrix_generation_domain_y,dtype=np.float32)
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

def rescale_linear(array):
	"""Rescale an arrary linearly."""
	# result_matrix=np.asarray(array)
	new_min=0
	new_max=255
	minimum, maximum = np.amin(array), np.amax(array)
	m = (new_max - new_min) / (maximum - minimum)
	b = new_min - m * minimum
	result_matrix = m * np.asarray(array) + b
	return result_matrix.astype(int) 

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

def concatenate(video_list):
	stringa = "ffmpeg -y -i \"concat:"
	elenco_file_temp = []
	for f in video_list:
		file = "img/temp" + str(video_list.index(f) ) + ".ts"
		os.system("ffmpeg -i " + f + " -c copy -bsf:v h264_mp4toannexb -f mpegts " + file)
		elenco_file_temp.append(file)
	# print(elenco_file_temp)
	for f in elenco_file_temp:
		stringa += f
		if elenco_file_temp.index(f) != len(elenco_file_temp)-1:
			stringa += "|"
		else:
			stringa += "\" -c copy  -bsf:a aac_adtstoasc output.mp4"
	# print(stringa)
	os.system(stringa)
 

if __name__ == "__main__":
	# OUTPUT_SIZE_IN_PIXELS_X = 1080 # number of columns
	# OUTPUT_SIZE_IN_PIXELS_Y = 1920 # number of rows
	OUTPUT_SIZE_IN_PIXELS_X = 1440 # number of columns
	OUTPUT_SIZE_IN_PIXELS_Y = 2560 # number of rows
	X_RANGE=1                    # range of y values 
	MAX_ITERATIONS = 90            # max number of iterations in single pixel calculation
	MANDELBROT_THRESHOLD = 2       # thresold of the absolute value of reiterated Z
	MIN=1000                       # start point of C values 
	MAX=1200                       # end point of C values
	SPEEDF = 0.1                   # speed of change of C value
	POWR=2                         # powr of Z in iteration function
	CX=0.1                         # position of x center
	CY=-0.55                       # position of y center
	CX=0.413238151606368892027     # position of y center
	CY=-1.24254013716898265806     # position of y center	
	MANDELBROT=1                   # 1 = mandelbrot set , 0 = julia set
	CYCLEFRAME=100	
	set_start_method("spawn")

	loops=MAX-MIN
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

	if loops>CYCLEFRAME:
		nrloops=loops//CYCLEFRAME		

	zoomnp=np.linspace(0,1, num=loops)
	counter=0
	ccycle=0
	video_list=[]
	for xcycle in range(nrloops):
		min=MIN+ccycle*CYCLEFRAME
		max=min+CYCLEFRAME
		result_matrix=[]
		for i in range (min,max):
			z=zoomnp[counter]
			result_matrix.append(opencl_ctx.run_julia(i,i/50,z))
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
		filename='img/julia'+str(ccycle)+'.mp4'
		p = Process(target=save_file,args=(filename,result_matrix,fig,ims,ccycle,figuresize_x,figuresize_y,))
		p.start()
		p.join()
		video_list.append(filename)
		del p
		del result_matrix
		# ani.save('img/julia'+str(ccycle)+'.mp4',fps=60,extra_args=["-threads", "4"])
		# ani.save('julia'+str(ccycle)+'.mp4',fps=15,extra_args=["-threads", "4","-codec","hevc"])
		ccycle+=1

	concatenate(video_list)	