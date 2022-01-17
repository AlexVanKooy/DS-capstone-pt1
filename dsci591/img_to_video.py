import numpy as np
from os.path import isfile, join
import cv2
import os

"""
This is the companion script to the Image_gen.py

specify the path of the images, ensuring they are named in the pattern `image#.jpg`
The video will be generated in the working directory named by the pathOut variable

"""
# Code was modified from https://medium.com/@iKhushPatel/convert-video-to-images-images-to-video-using-opencv-python-db27a128a481

pathIn= './no2_images/'
pathOut = 'video.avi'

# fps determines the number of images presented per second. 
fps = 5

frame_array = []
files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
#for sorting the file names properly
files.sort(key = lambda x: x[5:-4])
for i in range(len(files)):
    filename=pathIn + files[i]
    #reading each files
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    
    #inserting the frames into an image array
    frame_array.append(img)
    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
for i in range(len(frame_array)):
    out.write(frame_array[i])
out.release()

