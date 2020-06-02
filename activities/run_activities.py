# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 00:12:05 2020

@author: Kellin
"""


import argparse
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('video_id')
    parser.add_argument('camera_id')
    args = parser.parse_args()
    
    # Specify path
    input_path = args.path
    video_id = args.video_id
    camera_id = int(args.camera_id)
    
    import pandas as pd

    with open(input_path + '/output2.txt', 'r') as filehandle: 
      line_list = filehandle.readlines()
      
    
    line_list = [line.replace('[','') for line in line_list]
    line_list = [line.replace(']','') for line in line_list]
    line_list = [line.split() for line in line_list]
    
    
    df = pd.DataFrame(line_list,columns=['frame','person_id','bbox_left','bbox_top','bbox_width','bbox_height'])
    df.to_csv('output2.csv')
    
    output_df = pd.read_csv('output2.csv')
    output_df['height_corrected'] = output_df.bbox_height - output_df.bbox_top # bbox_height is mislabled, it's actually the bottom and not a height at all
    output_df.groupby('person_id')['height_corrected'].mean()
    
    
    # this function from https://stackoverflow.com/questions/28723670/intersection-over-union-between-two-detections
    
    def get_iou(bb1, bb2):
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.
    
        Parameters
        ----------
        bb1 : dict
            Keys: {'x1', 'x2', 'y1', 'y2'}
            The (x1, y1) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner
        bb2 : dict
            Keys: {'x1', 'x2', 'y1', 'y2'}
            The (x, y) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner
    
        Returns
        -------
        float
            in [0, 1]
        """
        assert bb1['x1'] < bb1['x2']
        assert bb1['y1'] < bb1['y2']
        assert bb2['x1'] < bb2['x2']
        assert bb2['y1'] < bb2['y2']
    
        # determine the coordinates of the intersection rectangle
        x_left = max(bb1['x1'], bb2['x1'])
        y_top = max(bb1['y1'], bb2['y1'])
        x_right = min(bb1['x2'], bb2['x2'])
        y_bottom = min(bb1['y2'], bb2['y2'])
    
        if x_right < x_left or y_bottom < y_top:
            return 0.0
    
        # The intersection of two axis-aligned bounding boxes is always an
        # axis-aligned bounding box
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
        # compute the area of both AABBs
        bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
        bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])
    
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
        assert iou >= 0.0
        assert iou <= 1.0
        return iou
    
    
    
    import cv2
    import glob
    
    video_path = glob.glob(input_path + '/*.avi')
    video_path.extend(glob.glob(input_path + '/*.mp4'))
    
    if len(video_path) > 1:
      raise Exception('Error: multiple videos not currently supported')
    
    cap = cv2.VideoCapture(video_path[0])
    #cap = cv2.VideoCapture("/content/gdrive/My Drive/Retail Analytics Data/Video_Set_1/ID_14/Camera_4/Seq_3/Seq_3.avi")
    
    frame_id = 1
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    clip_width = 192
    clip_height = 384
    #out = cv2.VideoWriter(input_path + "/full_pipeline_test.avi",fourcc, 15.0,(frame_width,frame_height))
    out = cv2.VideoWriter("{}.mp4".format(video_id),fourcc, 15.0,(frame_width,frame_height))
    
    data_list = []
    
    while True:
      print(frame_id)		
    
      ret,frame = cap.read()
      if ret is False:
        frame_id+=1
        break	
    
      for idx,row in output_df[output_df.frame == frame_id].iterrows():
        prev_row3 = output_df[(output_df.frame == row.frame - 3) & (output_df.person_id == row.person_id)]
        prev_row6 = output_df[(output_df.frame == row.frame - 6) & (output_df.person_id == row.person_id)]
        bbox_dict = {'x1': row.bbox_left, 'x2': row.bbox_width, 'y1': row.bbox_top, 'y2': row.bbox_height}
        if prev_row3.size > 0 and prev_row6.size > 0:
          bbox_dict3 = {'x1': prev_row3.bbox_left.values[0], 'x2': prev_row3.bbox_width.values[0], 'y1': prev_row3.bbox_top.values[0], 'y2': prev_row3.bbox_height.values[0]}
          bbox_dict6 = {'x1': prev_row6.bbox_left.values[0], 'x2': prev_row6.bbox_width.values[0], 'y1': prev_row6.bbox_top.values[0], 'y2': prev_row6.bbox_height.values[0]}
          iou03 = get_iou(bbox_dict,bbox_dict3)
          iou06 = get_iou(bbox_dict,bbox_dict6)
          iou36 = get_iou(bbox_dict3,bbox_dict6)
          avg_iou = (iou03 + iou06 + iou36) / 3
        elif prev_row3.size > 0:
          bbox_dict3 = {'x1': prev_row3.bbox_left.values[0], 'x2': prev_row3.bbox_width.values[0], 'y1': prev_row3.bbox_top.values[0], 'y2': prev_row3.bbox_height.values[0]}
          avg_iou = get_iou(bbox_dict,bbox_dict3)
        elif prev_row6.size > 0:
          bbox_dict6 = {'x1': prev_row6.bbox_left.values[0], 'x2': prev_row6.bbox_width.values[0], 'y1': prev_row6.bbox_top.values[0], 'y2': prev_row6.bbox_height.values[0]}
          avg_iou = get_iou(bbox_dict,bbox_dict6)
        else:
          avg_iou = 0 # no previous frames, assume they're moving
    
        # CORRECTION NEEDED: round instead of int()
        if camera_id == 0:
            if row.height_corrected < 125: # sitting (RED) (70 for 1st video, 125 for 2nd, 50 for 3rd)
              activity_flag = 0 
              cv2.rectangle(frame,(int(row.bbox_left),int(row.bbox_top)),(int(row.bbox_width),int(row.bbox_height)),(0,0,255),2)
            elif avg_iou > 0.9: # standing (BLUE)
              cv2.rectangle(frame,(int(row.bbox_left),int(row.bbox_top)),(int(row.bbox_width),int(row.bbox_height)),(255,0,0),2)
              activity_flag = 1 
            else: # moving (GREEN)
              cv2.rectangle(frame,(int(row.bbox_left),int(row.bbox_top)),(int(row.bbox_width),int(row.bbox_height)),(0,255,0),2)
              activity_flag = 2 
            cv2.putText(frame, str(int(row.person_id)),(int(row.bbox_left), int(row.bbox_top)),0, 5e-3 * 200, (0,0,0),2)
            data_list.append([row.frame,row.person_id,row.bbox_left,row.bbox_top,row.bbox_width,row.bbox_height,activity_flag,video_id, camera_id])
        elif camera_id == 1:
            if row.height_corrected < 70: # sitting (RED) (70 for 1st video, 125 for 2nd, 50 for 3rd)
              activity_flag = 0 
              cv2.rectangle(frame,(int(row.bbox_left),int(row.bbox_top)),(int(row.bbox_width),int(row.bbox_height)),(0,0,255),2)
            elif avg_iou > 0.9: # standing (BLUE)
              cv2.rectangle(frame,(int(row.bbox_left),int(row.bbox_top)),(int(row.bbox_width),int(row.bbox_height)),(255,0,0),2)
              activity_flag = 1 
            else: # moving (GREEN)
              cv2.rectangle(frame,(int(row.bbox_left),int(row.bbox_top)),(int(row.bbox_width),int(row.bbox_height)),(0,255,0),2)
              activity_flag = 2 
            cv2.putText(frame, str(int(row.person_id)),(int(row.bbox_left), int(row.bbox_top)),0, 5e-3 * 200, (0,0,0),2)
            data_list.append([row.frame,row.person_id,row.bbox_left,row.bbox_top,row.bbox_width,row.bbox_height,activity_flag,video_id, camera_id])
        else: print("Error: camera_id not recognized.", file=sys.stderr)
      out.write(frame)
      frame_id+=1
    
    final_df = pd.DataFrame(data_list, columns = ['frame','person_id','bbox_left','bbox_top','bbox_width','bbox_height', 'activity', 'video_id', 'camera_id'])
    final_df.to_csv(input_path + '/full_pipeline_output.csv', index=False)
    out.release()
    
    bash_command = "ffmpeg -i {}.mp4 -vcodec libx264 {}/{}.mp4 -y".format(video_id, input_path, video_id)
    import subprocess
    bash_result = subprocess.run(bash_command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,shell=True)
    '''
    import mysql.connector
    from mysql.connector import Error
    
    try:
        connection = mysql.connector.connect(host='35.226.231.168',
                                             database='output_data',
                                             user='root',
                                             password='5mb6aG2pn1FadH1N',
                                             allow_local_infile=True)
        if connection.is_connected():
            db_Info = connection.get_server_info()
            print("Connected to MySQL Server version ", db_Info)
            cursor = connection.cursor()
            cursor.execute("LOAD DATA LOCAL INFILE '{}'  \
                            INTO TABLE data \
                            FIELDS TERMINATED BY ',' \
                            IGNORE 1 LINES;".format(input_path + '/full_pipeline_output.csv'))
    
    except Error as e:
        print("Error while connecting to MySQL", e)
    finally:
        if (connection.is_connected()):
            connection.commit()
            cursor.close()
            connection.close()
            print("MySQL connection is closed")
    '''