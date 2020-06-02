# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 13:48:34 2020

@author: Kellin

Code to parse HDA person dataset (http://vislab.isr.ist.utl.pt/hda-dataset/) labels into YOLO input format. 
"""

# Specify path
input_path = '/home/kellinpelrine/bucket'

# Specify image dimensions
img_height = 480
img_width = 640

import re
import cv2

with open(input_path + '/cam02_edited2.txt') as f:
  text = f.read()

str_list = text.split()
start_list = []
end_list = []
for s in str_list:
  if bool(re.match(r'str=[0-9]+',s)):
    temp_str = s.split('=')
    start = temp_str[1]
    start_list.append(start)
  if bool(re.match(r'end=[0-9]+',s)):
    temp_str = s.split('=')
    end = temp_str[1]
    end_list.append(end)

start_list = [int(frame) for frame in start_list]
end_list = [int(frame) for frame in end_list]

cap = cv2.VideoCapture(input_path + '/camera02.avi')
i=0
write_flag = False
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    if i in start_list:
      write_flag += 1
    if write_flag > 0:
      cv2.imwrite('/home/kellinpelrine/darknet/build/darknet/x64/data/custom/frame_'+str(i)+'.jpg',frame)
    if i in end_list:
      write_flag -= 1
    i+=1
cap.release()  







# First clean up any existing files.
# This is necessary because later we append to files rather than writing them from scratch.
import os
for i in range(9500):
  if os.path.exists('/home/kellinpelrine/darknet/build/darknet/x64/data/custom/frame_{}.txt'.format(i)):
    os.remove('/home/kellinpelrine/darknet/build/darknet/x64/data/custom/frame_{}.txt'.format(i))

# Now parse all the bounding boxes and write them in YOLO retraining input format.
left_list = []
top_list = []
width_list = []
height_list = []
bbox_flag = False
cycle_counter = 0
person_counter = 0
for s in str_list:
  if bool(re.match(r'=\[[0-9.]+',s)):
    left_list.append(s.split('[')[1])
    bbox_flag = True
    cycle_counter = 1
    continue
  if bool(re.match(r'\]',s)):
    bbox_flag = False
    for i in range(end_list[person_counter] - start_list[person_counter] + 1):
      with open('/home/kellinpelrine/darknet/build/darknet/x64/data/custom/frame_' + str(start_list[person_counter] + i) + '.txt', 'a') as filehandle:
        left = float(left_list.pop(0))
        top = float(top_list.pop(0))
        width = float(width_list.pop(0))
        height = float(height_list.pop(0))

        # YOLO retraining requires bbox centers rather than top left corner.
        x_center = left + width/2
        y_center = top + height/2

        # Normalize
        x_center = x_center / img_width
        y_center = y_center / img_height
        width = width/img_width
        height = height/img_height
        filehandle.write("0 {} {} {} {}\n".format(x_center,y_center,width,height))
    person_counter += 1
    continue
  if bbox_flag:
    if cycle_counter == 0:
      left_list.append(s)
      cycle_counter = 1
      continue
    if cycle_counter == 1:
      top_list.append(s)
      cycle_counter = 2
      continue
    if cycle_counter == 2:
      width_list.append(s)
      cycle_counter = 3
      continue
    if cycle_counter == 3:
      height_list.append(s[:-1])
      cycle_counter = 0
      continue
  
    
import glob

# functions for sorting paths in natural way (using number in the filename)
def int_detector(text):
    return int(text) if text.isdigit() else text
def natural_sorter(text):
    return [ int_detector(c) for c in re.split('(\d+)',text) ]

frame_list = glob.glob('/home/kellinpelrine/darknet/build/darknet/x64/data/custom/' + '*.jpg')
frame_list.sort(key=natural_sorter)

frame_list = [path.split('/',4)[4] for path in frame_list]


import random
random.seed(43)
train = random.sample(frame_list,int(0.8*len(frame_list)))
test = [x for x in frame_list if x not in train]

with open('/home/kellinpelrine/darknet/build/darknet/x64/data/train.txt', 'w') as filehandle:
    filehandle.writelines("%s\n" % path for path in train)
with open('/home/kellinpelrine/darknet/build/darknet/x64/data/test.txt', 'w') as filehandle:
    filehandle.writelines("%s\n" % path for path in test)
