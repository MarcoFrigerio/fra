import tkinter as tk
import rgb_julia_ani5 as j

w = tk.Tk()
w.geometry=("800x800")
w.title("Loading Mask")
start_button=tk.Button(text="StartCalculation",command=(lambda: j.julia()))
start_button.grid(row=0,column=0)
tx1=tk.Entry()
tx1.grid(row=1,column=0)
tx1.label=j.s.X_RANGE.__str__
tx1.insert(0,j.s.X_RANGE)

if __name__ == "__main__":
	w.mainloop()