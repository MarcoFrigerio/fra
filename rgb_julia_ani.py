#!/usr/bin/python3


import PIL
import pyopencl as cl
import numpy as np
import os
import matplotlib.pyplot as plt
# from PIL import Image
import matplotlib.animation as animation

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

		# Kernel function instantiation
		self.prg = cl.Program(self.ctx, proc_src).build()


	def run_mandelbrot(self,printspeed=False):
		mandel_shape=(OUTPUT_SIZE_IN_PIXELS,OUTPUT_SIZE_IN_PIXELS,3)
		mf = cl.mem_flags# opencl memflag enum
		#matrix_generation_domain = np.linspace(-MANDELBROT_THRESHOLD, MANDELBROT_THRESHOLD, num=OUTPUT_SIZE_IN_PIXELS)
		matrix_generation_domain = np.linspace(-1.5, 1.5, num=OUTPUT_SIZE_IN_PIXELS)

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

def pycl_mandelbrot(platform,i,mt):
		opencl_ctx = opencl_py(platform,'julia')
		opencl_ctx.compile({"OUTPUT_SIZE_IN_PIXELS":str(OUTPUT_SIZE_IN_PIXELS),
							"MAX_ITERATIONS":str(i),
							"MANDELBROT_THRESHOLD":str(mt)})
		return opencl_ctx.run_mandelbrot()

# def render_mandelbrot(i,cmap,interpolation):
# 	matrix_mandel=pycl_mandelbrot(0,i,2)
# 	result_matrix = np.array(matrix_mandel)
# 	plt.axis("off")
# 	im=plt.imshow(result_matrix, cmap=cmap, interpolation=interpolation ,animated=True)
# 	ims.append([im])

if __name__ == "__main__":
	ims = []
	figuresize=OUTPUT_SIZE_IN_PIXELS/100
	fig=plt.figure(figsize=(figuresize, figuresize))
	fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
		
	for i in range (25,120):
		mt=i/40
		matrix_mandel=pycl_mandelbrot(0,i,mt)
		result_matrix = np.array(matrix_mandel)
		plt.axis("off")
		#plt.figure(figsize=(19.8, 10.8))
		im=plt.imshow(result_matrix,animated=True,interpolation="bilinear")
		ims.append([im])
	for i in range (120,25,-1):
		mt=i/40
		matrix_mandel=pycl_mandelbrot(0,i,mt)
		result_matrix = np.array(matrix_mandel)
		plt.axis("off")
		#plt.figure(figsize=(19.8, 10.8))
		im=plt.imshow(result_matrix,animated=True,interpolation="bilinear")
		ims.append([im])

	#ims.extend(ims.reverse())
	# cmap="jet", interpolation="bilinear",alpha=1
	ani = animation.ArtistAnimation(fig, ims, interval=100_000, blit=True,repeat_delay=100_000)
	#ani = animation.FuncAnimation(fig, ims,interval=0, blit=True, repeat=True)
	Writer = animation.writers['ffmpeg']
	writer = Writer(fps=15, metadata=dict(artist='Marco!'), bitrate=1800,extra_args=["-hwaccel", "cuda"])
	ani.save('julia.mp4', fps=15)
	#plt.show()
	#plt.savefig("mandelbrot.jpg",format='jpg', bbox_inches='tight', pad_inches=0)
	# PIL_image = Image.fromarray(result_matrix,"RGB")
	# PIL_image.show()
	# #png.from_array(a, mode="L").save("/tmp/foo.png")