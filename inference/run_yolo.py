# -*- coding: utf-8 -*-
"""
Note: description below is copied from Jupyter NB description; UPDATE NEEDED.

Prototype tracker, designed for tracking people in retail store video data.

This prototype uses 3 base algorithms. They have been modified to work together and to create a full person tracking algorithm:
1. YOLOv3, for person detection. See https://github.com/AlexeyAB/darknet and https://dev.to/kojikanao/yolo-on-google-colab-4b8e.
2. Deepsort, for initial person tracking. See https://github.com/abhyantrika/nanonets_object_tracking.
3. Reidentification algorithm, for improved person tracking, especially between cameras and without ID swaps. See https://github.com/layumi/Person_reID_baseline_pytorch.git.

To use:
1. Specify input_path in the cell below. This should be the directory where the video for tracking is located. It can be formatted as .avi or .mp4, or given as individual .jpg frames.
2. Specify framerate.
3. Comment the two Google Drive mounting lines in the cell below if not using that.
4. The third cell is optional, for emptying gallery/query folders (i.e. discarding the history of people the tracker has seen and identified).
5. Run the rest.

Besides the gallery folder, this will generate an output video in the location the input directory is stored, \<input directory name\>_tracked.avi
"""
from __future__ import print_function, division

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='input path')
    parser.add_argument('path')
    args = parser.parse_args()
    
    # Specify path
    input_path = args.path
    
    # Specify framerate.
    frame_rate = 15.0
    
    import os
    base_directory = os.path.split(input_path)[0] # for use constructing some paths
    
    
    # Optional cell to empty gallery/query, for a clean run (rebuilding from scratch the data on previously seen people).
    #import shutil
    
    #if os.path.isdir(base_directory + '/gallery'):  
    #  shutil.rmtree(base_directory + '/gallery') 
    #if os.path.isdir(base_directory + '/query'):
    #  shutil.rmtree(base_directory + '/query') 
    if not os.path.isdir(base_directory + '/gallery'):  
        os.mkdir(base_directory + '/gallery')
    if not os.path.isdir(base_directory + '/query'): 
        os.mkdir(base_directory + '/query')
    
    
    
    
    
    # Input for YOLO requires the video to be stored as individual frames.
    # Input for tracker requires a video file (currently).
    # Here we collect the file names from the given path, construct video from frames or frames from video if needed,
    # and process the list of frames into YOLO input format.
    
    import glob
    import cv2
    import re
    
    # functions for sorting paths in natural way (using number in the filename)
    def int_detector(text):
        return int(text) if text.isdigit() else text
    def natural_sorter(text):
        return [ int_detector(c) for c in re.split('(\d+)',text) ]
    
    
    frame_list = glob.glob(input_path + '/*.jpg')
    frame_list.sort(key=natural_sorter)
    frame_list = frame_list #[:300] # to process more quickly, we just take 300 frames here
    video_path = glob.glob(input_path + '/*.avi')
    video_path.extend(glob.glob(input_path + '/*.mp4'))
    
    if len(video_path) > 1:
      raise Exception('Error: multiple videos not currently supported')
    if not frame_list and not video_path:
      raise Exception('No images or video found')
    
    elif not frame_list: # found video but not individual frames
      cap = cv2.VideoCapture(video_path[0])
      i=0
      while(cap.isOpened()):
          ret, frame = cap.read()
          if ret == False:
              break
          cv2.imwrite(input_path + '/frame_'+str(i)+'.jpg',frame)
          i+=1
      cap.release()  
      frame_list = glob.glob(input_path + '/*.jpg')
      frame_list.sort(key=natural_sorter)
    
    elif not video_path: # found frames but not video
      img_array = []
      for filename in frame_list:
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img) 
      out = cv2.VideoWriter(input_path + '/video.avi',cv2.VideoWriter_fourcc(*'HFYU'), frame_rate, size)
      for i in range(len(img_array)):
          out.write(img_array[i])
      out.release() 
      video_path = sorted(glob.glob(input_path + '/*.avi'))
      
    os.chdir('/darknet')
    # write frame locations to txt file for YOLO input
    if os.path.exists('input_images.txt'):
      os.remove('input_images.txt') 
    with open('input_images.txt', 'w') as filehandle: 
        filehandle.writelines("%s\n" % path for path in frame_list)
        
        
        
    # Apply YOLO.
    # Note this may take some time, depending on amount of frames, and will not display YOLO output.

    YOLO_out_path = input_path + "/YOLO_result.txt"
    bash_command = "./darknet detector test cfg/coco.data cfg/yolov3.cfg yolov3.weights -dont_show -ext_output < \"input_images.txt\" > \"{}\"".format(YOLO_out_path)
    import subprocess
    yolo_bash_result = subprocess.run(bash_command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,shell=True)
    print(yolo_bash_result.stdout)
    
    
    # Parse YOLO results into tracker+reID input format
    
    
    with open(YOLO_out_path) as f:
        lines = [line.rstrip() for line in f]
    del lines[:4]
    del lines[-1]
    
    bbox_list = []
    counter = 0
    for line in lines:
      if line.startswith('Enter Image Path:'):
        m = re.search('frame_(\d+)', line, re.IGNORECASE)
        frame_id = m.group(1)
        counter += 1
        continue
      if line.startswith('person:'):
        bbox_dims = []
        m = re.search('left_x:[ -]+(\d+)', line, re.IGNORECASE)
        bbox_dims.append(m.group(1))
        m = re.search('top_y:[ -]+(\d+)', line, re.IGNORECASE)
        bbox_dims.append(m.group(1))
        m = re.search('width:[ -]+(\d+)', line, re.IGNORECASE)
        bbox_dims.append(m.group(1))
        m = re.search('height:[ -]+(\d+)', line, re.IGNORECASE)
        bbox_dims.append(m.group(1))
        m = re.search('person:[ -]+(\d+)', line, re.IGNORECASE)
        bbox_dims.append(m.group(1))
        new_line = str(frame_id) + ',-1,' + bbox_dims[0] + ',' + bbox_dims[1] + ',' + bbox_dims[2] + ',' + bbox_dims[3] + ',' + str(float(bbox_dims[4])/100) + ',-1,-1,-1'
        bbox_list.append(new_line)
    
    det_path = input_path + "/det_yolo3.txt"
    if os.path.exists(det_path):
      os.remove(det_path) 
    with open(det_path, 'w') as filehandle:
        filehandle.writelines("%s\n" % bbox for bbox in bbox_list)
        
        
    
    # Prepare for tracker+reID
    os.chdir('/nanonets_object_tracking/')
    bash_command = "cp \'{}\' 'det/det_yolo3.txt'".format(det_path)
    os.system(bash_command)
    


