from os import stat
import tkinter as tk
from tkinter.constants import DISABLED

import rgb_julia_ani5 as j
# from varname import nameof
import json


class init_w:
	def __init__(self,f):
		self.f=f
		with open(self.f) as f:
			self.params=json.load(f)

		self.w = tk.Tk()
		self.w.geometry=("800x800")
		self.w.title("Loading Mask")
		self.myentry=[]
		self.mylabel=[]
		i=0
		for key in self.params:
			self.myentry.append(tk.Entry(self.w))
			self.myentry[i].insert(0,self.params[key])
			self.mylabel.append(tk.Label(self.w,text=key))
			self.myentry[i].grid(row=i,column=1)
			self.mylabel[i].grid(row=i,column=0)
			i+=1

		self.start_button=tk.Button(self.w,text="StartCalculation",command=lambda:j.julia(),state=DISABLED)
		self.start_button.grid(row=i+2,column=0)
		self.validate_button=tk.Button(self.w,text="Validate Values",command=self.validate_values)
		self.validate_button.grid(row=i+1,column=0)

	def validate_values(self):
		i=0
		for key in  self.params:
			self.params[key]=self.myentry[i].get()
			i+=1
		with open(self.f,'w') as f:
			json.dump(self.params,f,indent=1)
		
		self.start_button.configure(state='active')
	

if __name__ == "__main__":
	
	f="julia_parm.json"

	w=init_w(f)
	w.w.mainloop()