#!/usr/bin/python3


import PIL
import pyopencl as cl
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image

OUTPUT_SIZE_IN_PIXELS = 11_000
MAX_ITERATIONS = 80
MANDELBROT_THRESHOLD = 1.8

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

		# Kernel function instantiation
		self.prg = cl.Program(self.ctx, proc_src).build()


	def run_mandelbrot(self,printspeed=False):
		mandel_shape=(OUTPUT_SIZE_IN_PIXELS,OUTPUT_SIZE_IN_PIXELS,3)
		mf = cl.mem_flags# opencl memflag enum
		matrix_generation_domain = np.linspace(-MANDELBROT_THRESHOLD, MANDELBROT_THRESHOLD, num=OUTPUT_SIZE_IN_PIXELS)

		gD_np = np.array(matrix_generation_domain,dtype=np.float32)
		gD_g = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=gD_np)

		result = np.empty(mandel_shape, dtype=np.uint32)
		result_g = cl.Buffer(self.ctx, mf.WRITE_ONLY,result.nbytes)# size should be in byte
		
		start_event=cl.enqueue_marker(self.queue)

		finish_event=self.prg.mandelbrot(self.queue,
			mandel_shape,
			None ,
			gD_g,result_g )
		finish_event.wait()

		rt = cl.enqueue_copy(self.queue, result, result_g)


		return result

	def run_mandelbrot_loop(self,printspeed=False):
		mandel_shape=(OUTPUT_SIZE_IN_PIXELS,OUTPUT_SIZE_IN_PIXELS,3)
		mf = cl.mem_flags# opencl memflag enum
		matrix_generation_domain = np.linspace(-MANDELBROT_THRESHOLD, MANDELBROT_THRESHOLD, num=OUTPUT_SIZE_IN_PIXELS)

		gD_np = np.array(matrix_generation_domain,dtype=np.float32)
		gD_g = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=gD_np)

		result = np.empty(mandel_shape, dtype=np.uint32)
		result_g = cl.Buffer(self.ctx, mf.WRITE_ONLY,result.nbytes)# size should be in byte
		
		start_event=cl.enqueue_marker(self.queue)

		finish_event=self.prg.mandelbrot(self.queue,
			mandel_shape,
			None ,
			gD_g,result_g )
		finish_event.wait()

		rt = cl.enqueue_copy(self.queue, result, result_g)


def pycl_mandelbrot(platform):
		opencl_ctx = opencl_py(platform,'julia')
		opencl_ctx.compile({"OUTPUT_SIZE_IN_PIXELS":str(OUTPUT_SIZE_IN_PIXELS),
							"MAX_ITERATIONS":str(MAX_ITERATIONS),
							"MANDELBROT_THRESHOLD":str(MANDELBROT_THRESHOLD)})
		return opencl_ctx.run_mandelbrot()	

if __name__ == "__main__":
	matrix_mandel=pycl_mandelbrot(0)
	print(matrix_mandel.shape)
	# print(matrix_mandel[400][400])
	result_matrix = np.array(matrix_mandel)
	#plt.figure(figsize=(19.8, 10.8))
	figuresize=OUTPUT_SIZE_IN_PIXELS/100
	fig=plt.figure(figsize=(figuresize, figuresize))
	plt.axis("off")
	plt.imshow(result_matrix) 
	# cmap="jet", interpolation="bilinear",alpha=1
	
	#plt.show()
	plt.savefig("julia.jpg",format='jpg', bbox_inches='tight', pad_inches=0)
	# PIL_image = Image.fromarray(result_matrix,"RGB")
	# PIL_image.show()
	# #png.from_array(a, mode="L").save("/tmp/foo.png")
