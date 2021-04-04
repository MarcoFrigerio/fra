# fractals

julia set OPENCL calulation 
you can set all this parameters

	OUTPUT_SIZE_IN_PIXELS_X = 1440  # number of columns
	OUTPUT_SIZE_IN_PIXELS_Y = 2560  # number of rows
	X_RANGE=0.0005                    # range of y values 
	MAX_ITERATIONS = 90             # max number of iterations in single pixel opencl calculation
	MANDELBROT_THRESHOLD = 2        # thresold of the absolute value of reiterated Z
	MIN=1                       # start point of C values 
	MAX=121                        # end point of C values
	SPEEDF = 0.1                    # speed of change of C value in julia set
	POWR=2                          # powr of Z in iteration function
	CX=0.01                          # position of x center (good for julia set)
	CY=-0.55                        # position of y center (good for julia set)
	CX=0.413238151606368892027      # position of y center (good for mandelbrot set)
	CY=-1.24254013716898265806      # position of y center	 (good for mandelbrot set)
	# CX=0      # position of y center
	# CY=0      # position of y center	
	MANDELBROT=1                    # 1 = mandelbrot set , 0 = julia set
	FLAG_ZOOM=True                  # Flag Zoom the image
	FRAMEEVERY=10                   # number of frames not calculated between two calculated
	COMPLEX_CAL=True                 # calculation with custom complex opencl definition
