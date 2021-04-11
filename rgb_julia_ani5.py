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
	# # OUTPUT_SIZE_IN_PIXELS_X = 1080 # number of columns
	# # OUTPUT_SIZE_IN_PIXELS_Y = 1920 # number of rows
	# OUTPUT_SIZE_IN_PIXELS_X = 1440  # number of columns
	# OUTPUT_SIZE_IN_PIXELS_Y = 2560  # number of rows
	# X_RANGE=1                   # initial start range of y values 
	# # MAX_ITERATIONS = 90             # max number of iterations in single pixel opencl calculation
	# MAX_ITERATIONS = 2_000             # max number of iterations in single pixel opencl calculation
	# MANDELBROT_THRESHOLD = 2        # thresold of the absolute value of reiterated Z=
	# MIN=1                       # start point of C values 
	# # MAX=70_000_000_000                        # end point of C values
	# # FRAMEEVERY=4_000_000                   # number of frames not calculated between two calculated
	# # MAX=1_400_000                        # end point of C values
	# # FRAMEEVERY=200                   # number of frames not calculated between two calculated
	# MAX=1_400_000                        # end point of C values
	# FRAMEEVERY=20_000                   # number of frames not calculated between two calculated
	# CYCLEFRAMEBASE=60
	# CYCLEFRAME=CYCLEFRAMEBASE*FRAMEEVERY
	# SPEEDF = 0.1                    # speed of change of C value in julia set
	# POWR=2                          # powr of Z in iteration function
	# CX=0.01                          # position of x center (good for julia set)
	# CY=-0.55                        # position of y center (good for julia set)
	# CX=np.float128(0.413238151606368892027)      # position of y center (good for mandelbrot set)
	# CY=np.float128(-1.24254013716898265806)      # position of y center	 (good for mandelbrot set)
	# CX = 0.1374168856037867 
	# CY = -0.7746806106269039
	# # CY = -0.7746806106269039		
	# # CX = 0.1374168856037867
	# # CX=-0.6413130610648031748603750151793020665794949522823052595561775430644485741727536902556370230689681162370740565537072149790106973211105273740851993394803287437606238596262287731075999483940467161288840614581091294325709988992269165007394305732683208318834672366947550710920088501655704252385244481168836426277052232593412981472237968353661477793530336607247738951625817755401065045362273039788332245567345061665756708689359294516668271440525273653083717877701237756144214394870245598590883973716531691124286669552803640414068523325276808909040317617092683826521501539932397262012011082098721944643118695001226048977430038509470101715555439047884752058334804891389685530946112621573416582482926221804767466258346014417934356149837352092608891639072745930639364693513216719114523328990690069588676087923656657656023794484324797546024248328156586471662631008741349069961493817600100133439721557969263221185095951241491408756751582471307537382827924073746760884081704887902040036056611401378785952452105099242499241003208013460878442953408648178692353788153787229940221611731034405203519945313911627314900851851072122990492499999999999999999991
	# # CY=0.360240443437614363236125244449545308482607807958585750488375814740195346059218100311752936722773426396233731729724987737320035372683285317664532401218521579554288661726564324134702299962817029213329980895208036363104546639698106204384566555001322985619004717862781192694046362748742863016467354574422779443226982622356594130430232458472420816652623492974891730419252651127672782407292315574480207005828774566475024380960675386215814315654794021855269375824443853463117354448779647099224311848192893972572398662626725254769950976527431277402440752868498588785436705371093442460696090720654908973712759963732914849861213100695402602927267843779747314419332179148608587129105289166676461292845685734536033692577618496925170576714796693411776794742904333484665301628662532967079174729170714156810530598764525260869731233845987202037712637770582084286587072766838497865108477149114659838883818795374195150936369987302574377608649625020864292915913378927790344097552591919409137354459097560040374880346637533711271919419723135538377394364882968994646845930838049998854075817859391340445151448381853615103761584177161812057928
	# DIR="img/"                   # working dir
	# # CX=math.e/20
	# # CY=math.e/7
	# # CX=0      # position of y center
	# # CY=0      # position of y center	
	# MANDELBROT=1                    # 1 = mandelbrot set , 0 = julia set
	# FLAG_ZOOM=True                  # Flag Zoom the image
	# COMPLEX_CAL=True                 # calculation with custom complex opencl definition

	set_start_method("spawn")

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
