import os

def concatenate(dirb,video_list):
	elenco_file_temp = []
	for f in video_list:
		if f.endswith(".mp4"):
			file = dirb+"temp" + str(video_list.index(f)).rjust(5,'0')  + ".ts"
			os.system("ffmpeg -y -i " + dirb+f + " -c copy -bsf:v h264_mp4toannexb -f mpegts " + file)
			elenco_file_temp.append(file)
	# print(elenco_file_temp)
	stringa = "ffmpeg -y -i \"concat:"
	# for f in elenco_file_temp:
	# 	stringa += f
	# 	# if elenco_file_temp.index(f) != len(elenco_file_temp)-1:
		# 	stringa += "|"
		# else:
		# 	stringa += "\" -c copy  -bsf:a aac_adtstoasc output.mp4"
		# for f in elenco_file_temp:
	input_file_list=dirb+"input_file_list"
	with open(input_file_list,"w") as f:
		for fts in elenco_file_temp:
			f.write("file '"+fts.lstrip(dirb)+"'\n")

	stringa = "ffmpeg -y -f concat -safe 0 -i "+input_file_list+" -c copy output.mp4 "

	# print(stringa)
	os.system(stringa)

if __name__ =="__main__":
	DIR="img/"
	video_list=os.listdir(DIR)
	video_list.sort()
	concatenate(DIR,video_list)
