from tkinter import *
from tkinter import filedialog
import tkinter as tk
from tkinter import ttk
import sv_ttk
from PIL import Image, ImageTk
import cv2
from matplotlib import pyplot as plt

import matplotlib.image as mpimg


import os
import numpy as np



def processWheel(event):
  global img1
  global img2
  global File1
  global File2
  global size
  global last_size


  if event.delta > 0:
    print("shang"+str(event.delta))
  else:
    print("xia"+str(event.delta))# 滚轮往下滚动，缩小
  last_size = size
  size = (size+int(event.delta)/2400)
  if size > 1:
    size = 1
  if size < 0.1:
    size = 0.1
  print(size)
  
def xFunc1(event):
  global img1
  global img2
  global File1
  global File2
  global x
  global y
  global size

  print(f"鼠标左键滑动坐标是:x={event.x},y={event.y}")

  x = event.x-int(300*size/2)# int(int(ev)*(300-(300*size))/10)#秀，不谈，乱秀
  if x > 300-300*size-1:
    x = 300-300*size-1
  if x < 0:
    x = 0
  y = event.y-int(225*size/2)# int(int(ev)*(225-(225*size))/10)#秀，不谈，乱秀
  if y > 225-225*size-1:
    y = 225-225*size-1
  if y < 0:
    y = 0
  x = int(x)
  y = int(y)


def Keyboard_down(event):
  print("down press")


if __name__ == "__main__": 
  canvasH = 240
  canvesW = 320
  canvesBD = 5
  root = Tk()#setting up a tkinter canvas with scrollbars
  sv_ttk.set_theme("dark")
  root.title("auto-Occlution")
  frame = Frame(root, bd=2, relief=SUNKEN)
  frame.grid_rowconfigure(0, weight=1)
  canvas = Canvas(frame, bd=canvesBD,bg = "white",height = canvasH,width = canvesW)#, xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
  canvas.grid(row=0, column=0, sticky=N+S+E+W)
  frame.pack(fill=BOTH,expand=1)
  
  frame2 = Frame(root, bd=canvesBD, relief=SUNKEN)
  frame2.grid_rowconfigure(0, weight=1)
  frame2.grid_columnconfigure(0, weight=1)
  frame2.pack()

  canvas2 = Canvas(frame, bd=canvesBD,bg = "white",height = canvasH,width = canvesW)#, xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
  canvas2.grid(row=1, column=0, sticky=N+S+E+W)

  canvas11 = Canvas(frame, bd=canvesBD,bg = "white",height = canvasH,width = canvesW)#, xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
  canvas11.grid(row=0, column=1, sticky=N+S+E+W)
  canvas22 = Canvas(frame, bd=canvesBD,bg = "white",height = canvasH,width = canvesW)#, xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
  canvas22.grid(row=1, column=1, sticky=N+S+E+W)
  
  canvas13 = Canvas(frame, bd=canvesBD,bg = "white",height = canvasH,width = canvesW)#, xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
  canvas13.grid(row=0, column=2, sticky=N+S+E+W)
  canvas23 = Canvas(frame, bd=canvesBD,bg = "white",height = canvasH,width = canvesW)#, xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
  canvas23.grid(row=1, column=2, sticky=N+S+E+W)

  # canvas24 = Canvas(frame, bd=canvesBD,bg = "white",height = canvasH,width = canvesW)#, xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
  # canvas24.grid(row=1, column=3, sticky=N+S+E+W)
  #绑定鼠标键盘事件
  canvas23.bind("<MouseWheel>", processWheel)
  canvas23.bind("<B1-Motion>", xFunc1)
  root.bind("<KeyPress-Down>",Keyboard_down)

  def printcoords_COI():
   global File1
   File1 = filedialog.askopenfilename(parent=root, initialdir="D:/LZ/1graduation/picture/",title='Choose an image.')
   print(File1)
   
   File1_resizing = cv2.imread(File1)
   File1_resizing = cv2.resize(File1_resizing,(int(canvesW),int(canvasH)))
   cv2.imwrite('picture/file1_resized.jpg',File1_resizing)
   File1 = 'picture/file1_resized.jpg'

   filename = ImageTk.PhotoImage(Image.open(File1))
   canvas.image = filename # <--- keep reference of your image
   canvas.create_image(0,0,anchor='nw',image=filename)
  Button(frame2,text='choose_COI',command=printcoords_COI).grid(row = 2,column = 0)

  def printcoords_BG():
    global File2
    File2 = filedialog.askopenfilename(parent=root, initialdir="D:/LZ/1graduation/picture/",title='Choose an image.')
    # checkRGBimg=mpimg.imread(File2)
    checkRGBimg=cv2.imread(File2)
    print('img_shape',checkRGBimg.shape)
    if checkRGBimg.shape[2]>3:
      checkRGBimg = Image.open(File2)
      save_img = checkRGBimg.convert('RGB')
      save_img.save('picture/32to24.jpg')
      File2_resizing = cv2.imread('picture/32to24.jpg')
      File2_resizing = cv2.resize(File2_resizing,(int(canvesW),int(canvasH)))
      # File2_resizing = cv2.resize(File2_resizing,(int(300*size),int(225*size)))
    else:
      File2_resizing = cv2.resize(checkRGBimg,(int(canvesW),int(canvasH)))
    cv2.imwrite('picture/file2_resized.jpg',File2_resizing)
    File2 = 'picture/file2_resized.jpg'
    filename = ImageTk.PhotoImage(Image.open(File2))
    # print(File2)
    #边缘
    # img = cv2.imread(File1)
    # edges = cv2.Canny(img,100,200)
    # img2 = cv2.imread(File2)
    # edges2 = cv2.Canny(img2,100,200)
    # plt.subplot(121),plt.imshow(edges,cmap = 'gray')
    # plt.title('Edge Image1'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122),plt.imshow(edges2,cmap = 'gray')
    # plt.title('Edge Image2'), plt.xticks([]), plt.yticks([])
    # plt.show()
    canvas2.image = filename # <--- keep reference of your image
    canvas2.create_image(0,0,anchor='nw',image=filename)
  Button(frame2,text='choose_BG',command=printcoords_BG).grid(row = 2,column = 1)

  
  root.mainloop()