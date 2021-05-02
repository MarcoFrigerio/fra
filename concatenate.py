import subprocess
import os
from colorama import Fore	

def exec_string(stringa,dirb,fileout,fileerr):
	with open(dirb+fileout,"w") as fout:
		with open(dirb+fileerr,"w") as ferr:
			out=subprocess.run([stringa],stdout=fout,stderr=ferr,shell=True)
			return(out.returncode)

def concatenate(dirb,video_list):
	logdir=dirb+"logs/"
	try:os.mkdir(logdir)
	except:pass
	elenco_file_temp = []
	for f in video_list:
		if f.endswith(".mp4"):
			findex=str(video_list.index(f)).rjust(5,'0')
			file = dirb+"temp" + findex  + ".ts"
			stringa="ffmpeg -y -i " + dirb+f + " -c copy -bsf:v h264_mp4toannexb -f mpegts " + file
			out=exec_string(stringa,logdir,"video_out_stout_"+findex+".log","video_out_sterr_"+findex+".log")
			if out!=0:
				print(f'{Fore.RED}ERROR WORKING {file} {Fore.RESET}')
			else:elenco_file_temp.append(file)
	# print(elenco_file_temp)
	stringa = "ffmpeg -y -i \"concat:"

	input_file_list=dirb+"input_file_list"
	with open(input_file_list,"w") as f:
		for fts in elenco_file_temp:
			f.write("file '"+fts.lstrip(dirb)+"'\n")

	stringa = "ffmpeg -y -f concat -safe 0 -i "+input_file_list+" -c copy output.mp4 "

	# print(stringa)
	out=exec_string(stringa,logdir,"video_all_stout.log","video_all_sterr.log")
	return out

if __name__ =="__main__":
	DIR="img/"
	video_list=os.listdir(DIR)
	video_list.sort()
	concatenate(DIR,video_list)
