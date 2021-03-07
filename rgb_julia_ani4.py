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
from multiprocessing import get_context

OUTPUT_SIZE_IN_PIXELS = 1500
MAX_ITERATIONS = 80
MANDELBROT_THRESHOLD = 2

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

	def compile(self,marcos=dict,writeProcessedOpenCLCode=False):
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
		julia_shape=(OUTPUT_SIZE_IN_PIXELS,OUTPUT_SIZE_IN_PIXELS,3)
		mf = cl.mem_flags# opencl memflag enum
		#matrix_generation_domain = np.linspace(-MANDELBROT_THRESHOLD, MANDELBROT_THRESHOLD, num=OUTPUT_SIZE_IN_PIXELS)
		matrix_generation_domain = np.linspace(-1.5, 1.5, num=OUTPUT_SIZE_IN_PIXELS)

		gD_np = np.array(matrix_generation_domain,dtype=np.float32)
		gD_g = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=gD_np)

		input_ib=np.int32(input_i)
		input_thre=np.float32(thre)

		result = np.empty(julia_shape, dtype=np.uint32)
		result_g = cl.Buffer(self.ctx, mf.WRITE_ONLY,result.nbytes)# size should be in byte
		
		start_event=cl.enqueue_marker(self.queue)
		
		finish_event=self.prg.julia(self.queue,
			julia_shape,
			None ,
			gD_g,
			input_ib,
			input_thre,
			result_g )
		finish_event.wait()
		
		rt = cl.enqueue_copy(self.queue, result, result_g)
		gD_g.release()
		result_g.release()
		return result


if __name__ == "__main__":
	set_start_method("spawn")

	ims = []
	
	figuresize=OUTPUT_SIZE_IN_PIXELS/100
	fig=plt.figure(figsize=(figuresize, figuresize))
	fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)


	MIN=1
	MAX=500
	loops=MAX-MIN
	STEPS=os.cpu_count()
	counter=0

	opencl_ctx = opencl_py(0,'julia')
	opencl_ctx.compile({"OUTPUT_SIZE_IN_PIXELS":str(OUTPUT_SIZE_IN_PIXELS)})
	
	
	result_matrix=[]
	for i in range (MIN,MAX):
		result_matrix.append(opencl_ctx.run_julia(i,i/50))
		#f_matrix_gen((opencl_ctx,i,i/40)))

	nr_im=len(result_matrix)
	im = []
	for i in range(nr_im):
		# im[i].write_png("img/test"+str(i)+".png")
		im.append(plt.imshow(result_matrix[i],animated=True,interpolation="bilinear"))
		ims.append([im[i]])




	# plt.savefig("img/julia.jpg",format='jpg', bbox_inches='tight', pad_inches=0)
	print("END LOOP...SAVING FILE....")
	try:
		plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
	except:
		pass
	plt.axis("off")	
	ani = animation.ArtistAnimation(fig, ims, interval=0, blit=True,repeat_delay=0,repeat=True)
	# ani.to_html5_video(0)
	# Writer = animation.writers['ffmpeg']
	# writer = Writer(fps=15, metadata=dict(artist='Marco!'), bitrate=1800,extra_args=["-hwaccel", "cuda"])
	# ani.save('julia.gif',writer='pillow',fps=15)
	# ani.save('julia.mp4',fps=60,extra_args=["-hwaccel", "cuda"])
	ani.save('julia.mp4',fps=60)
