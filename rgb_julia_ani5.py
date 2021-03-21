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


	def run_julia(self,input_i,thre,printspeed=False):
		julia_shape=(OUTPUT_SIZE_IN_PIXELS_X,OUTPUT_SIZE_IN_PIXELS_Y,4)
		mf = cl.mem_flags# opencl memflag enum
		# matrix_generation_domain = np.linspace(-MANDELBROT_THRESHOLD, MANDELBROT_THRESHOLD, num=OUTPUT_SIZE_IN_PIXELS)
		screen_format=OUTPUT_SIZE_IN_PIXELS_Y/OUTPUT_SIZE_IN_PIXELS_X
		x_range=X_RANGE
		y_range=x_range*screen_format
		matrix_generation_domain_x = np.linspace(-x_range, x_range, num=OUTPUT_SIZE_IN_PIXELS_X)
		matrix_generation_domain_y = np.linspace(-y_range, y_range, num=OUTPUT_SIZE_IN_PIXELS_Y)

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
			None ,
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

if __name__ == "__main__":
	set_start_method("spawn")
	OUTPUT_SIZE_IN_PIXELS_X = 1080
	OUTPUT_SIZE_IN_PIXELS_Y = 1920
	X_RANGE=1.3
	MAX_ITERATIONS = 80
	MANDELBROT_THRESHOLD = 2

	MIN=1
	MAX=800
	POWR=3

	ims = []

	loops=MAX-MIN
	opencl_ctx = opencl_py(0,'julia')
	opencl_ctx.compile({"OUTPUT_SIZE_IN_PIXELS_X":str(OUTPUT_SIZE_IN_PIXELS_X),
						"OUTPUT_SIZE_IN_PIXELS_Y":str(OUTPUT_SIZE_IN_PIXELS_Y),
						"POWR":str(POWR)})

	figuresize_y=OUTPUT_SIZE_IN_PIXELS_X/100
	figuresize_x=OUTPUT_SIZE_IN_PIXELS_Y/100
	fig=plt.figure(figsize=(figuresize_x, figuresize_y))

	result_matrix=[]
	for i in range (MIN,MAX):
		result_matrix.append(opencl_ctx.run_julia(i,i/50))
		#f_matrix_gen((opencl_ctx,i,i/40)))

	# print("RESCALING RGB VALUES..")
	# with Pool() as p:
	# 	result_matrix=p.map(rescal	e_linear,result_matrix)
	# print("BEGIN PLOTTING IMAGING..")
	nr_im=len(result_matrix)
	iml = []
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
	# print("SAVING...")
	# iml[0].save('anitest.gif',
	# 			save_all=True,
	# 			append_images=iml[1:],
	# 			duration=100,
	# 			loop=0)
	print("END LOOP...SAVING FILE....")
	try:
		plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
	except:
		pass
	plt.axis("off")
	

	fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

	ani = animation.ArtistAnimation(fig, ims, interval=0, blit=True,repeat_delay=0,repeat=True)
	ani.save('julia.mp4',fps=60,extra_args=["-threads", "4"])
	