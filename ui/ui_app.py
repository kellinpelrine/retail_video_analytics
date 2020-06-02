'''
Flask UI, inspired by https://github.com/lucko515/image-search-engine 
'''

#Import dependencies
import os

import io
import random
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import cv2
from copy import deepcopy as copy
import numpy as np
import shutil
import sys

import mysql.connector
from mysql.connector import Error

import subprocess

#import Flask dependencies
from flask import Flask, request, render_template, send_from_directory

# load db config
import yaml
with open("db_info.yaml", 'r') as f:
    try:
        db_config = yaml.safe_load(f)
        db_config = {key : value[0] for key, value in db_config.items()}
    except:
        print('db config missing')


#Set root dir
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

#Define Flask app
app = Flask(__name__, static_url_path='/static')

#Define apps home page
@app.route("/") #www.image-search.com/
def index():
	return render_template("index.html")

#Define upload function
@app.route("/upload", methods=["POST"])
def upload():

    upload_dir = os.path.join(APP_ROOT, "uploads")

    if not os.path.isdir(upload_dir):
        os.mkdir(upload_dir)
        
    id_list = []
    camera_id = 0
    
    for vid in request.files.getlist("file[]"):
        vid_name = vid.filename
        tmp_name = os.path.splitext(vid_name)[0]
        if 'camera0' in tmp_name:
            camera_id = 0
        elif 'camera1' in tmp_name:
            camera_id = 1
        else:
            print('Camera ID missing, using default 0', file=sys.stderr)
            
        if 'video1' in tmp_name:
            timestamp=8
        elif 'video2' in tmp_name:
            timestamp=10
        elif 'video3' in tmp_name:
            timestamp=12
        elif 'video4' in tmp_name:
            timestamp = 14
        elif 'video5' in tmp_name:
            timestamp = 16
        else:
            timestamp = 0
            print('Time missing, using default 0', file=sys.stderr)
            
        destination_dir = "/".join([upload_dir, tmp_name])
        print(destination_dir, file=sys.stderr)
        if not os.path.isdir(destination_dir):
            os.mkdir(destination_dir)
        destination = "/".join([upload_dir, os.path.splitext(vid_name)[0], vid_name])
        vid.save(destination)
        
        max_id = 0
        
        try:
            connection = mysql.connector.connect(host=db_config['host'],
                                                 database='output_data',
                                                 user=db_config['user'],
                                                 password=db_config['password'],
                                                 allow_local_infile=True)
            if connection.is_connected():
                db_Info = connection.get_server_info()
                print("Connected to MySQL Server version ", db_Info)
                cursor = connection.cursor()
                cursor.execute("SELECT MAX(video_id) \
                               FROM data;")
                
                tmp_max_id = cursor.fetchone()[0]
                if tmp_max_id is not None:
                    max_id += tmp_max_id + 100
        
        except Error as e:
            print("Error while connecting to MySQL", e)
        finally:
            if (connection.is_connected()):
                connection.commit()
                cursor.close()
                connection.close()
            print("MySQL connection is closed")
	
        bash_command = 'curl \"127.0.0.1:8082/run?video_id={}&camera_id={}&path={}\"'.format(max_id, camera_id, os.path.splitext(vid_name)[0])
        result = subprocess.run(bash_command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,shell=True)
        
        shutil.copy('uploads/' + os.path.splitext(vid_name)[0] + '/{}.mp4'.format(max_id), 'static/{}.mp4'.format(max_id))
        print('uploads/' + os.path.splitext(vid_name)[0] + '/{}.mp4'.format(max_id), file=sys.stderr)
        
        try:
            connection = mysql.connector.connect(host=db_config['host'],
                                                 database='output_data',
                                                 user=db_config['user'],
                                                 password=db_config['password'],
                                                 allow_local_infile=True)
            if connection.is_connected():
                db_Info = connection.get_server_info()
                print("Connected to MySQL Server version ", db_Info)
                cursor = connection.cursor()
                cursor.execute("LOAD DATA LOCAL INFILE '{}'  \
                                INTO TABLE data \
                                FIELDS TERMINATED BY ',' \
                                IGNORE 1 LINES \
                                (frame,person_id,bbox_left,bbox_top,bbox_width,bbox_height,activity,video_id,camera_id) \
                                SET timestamp = {} \
                                ;".format('uploads/' + os.path.splitext(vid_name)[0] + '/full_pipeline_output.csv', timestamp))    
                
                
        
        except Error as e:
            print("Error while connecting to MySQL", e)
        finally:
            if (connection.is_connected()):
                connection.commit()
                cursor.close()
                connection.close()
            print("MySQL connection is closed")
        id_list.append(max_id)

    #return result.stdout.decode('utf-8')
    #return Response(output.getvalue(), mimetype='image/png')
    #return "upload complete"
    return "Upload Complete. Video ID numbers: {}. \n Access individual video at http://34.74.59.41:5000/dashboard?video_id=video_id_number".format(id_list)

@app.route("/dashboard")
def display_dashboard():
    video_id = request.args.get('video_id', default = 0)
    
    connection = mysql.connector.connect(host=db_config['host'],
                                            database='output_data',
                                            user=db_config['user'],
                                            password=db_config['password'],
                                            allow_local_infile=True)
    if connection.is_connected():
        db_Info = connection.get_server_info()
        print("Connected to MySQL Server version ", db_Info)
        cursor = connection.cursor()
        cursor.execute("SELECT COUNT(DISTINCT(person_id)) \
                        FROM data;")
        total_people = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT(person_id)) \
                        FROM data \
                        WHERE camera_id = 0;")
        total_people0 = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT(person_id)) \
                        FROM data \
                        WHERE camera_id = 1;")
        total_people1 = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT(person_id)) \
                        FROM data \
                        WHERE activity = 0 \
                        GROUP BY person_id \
                        having COUNT(*) >= 15;")
        
        total_people_sitting = 0
        for item in cursor.fetchall():
            total_people_sitting += 1
        
        cursor.execute("SELECT person_id, COUNT(frame) FROM data GROUP BY person_id;")
        result = cursor.fetchall()
        
        x = [item[0] for item in result]
        height = [item[1] for item in result]
        fig0, ax0 = plt.subplots()
        ax0.bar(np.arange(len(x)), height, tick_label=[item[0] for item in result])
        ax0.set_xlabel('Person ID')
        ax0.set_ylabel('Time in store')
        #ax1.set_xticks([item[0] for item in result])
        fig0.savefig('static/plot.png')
        
        cursor.execute("SELECT COUNT(DISTINCT(person_id)), timestamp FROM data GROUP BY timestamp;")
        result = cursor.fetchall()
        
        x = [item[1] for item in result]
        height = [item[0] for item in result]
        fig1, ax1 = plt.subplots()
        ax1.bar(np.arange(len(x)), height, tick_label=[item[1] for item in result])
        ax1.set_xlabel('Time of day')
        ax1.set_ylabel('Number of customers')
        #ax1.set_xticks([item[0] for item in result])
        fig1.savefig('static/customers_by_time.png')
        
        fig2, ax2 = plt.subplots()
        
        base_img0 = cv2.imread('static/camera0_empty.jpg',1)
        base_img0 = cv2.cvtColor(base_img0, cv2.COLOR_BGR2GRAY)
        base_img0 = cv2.cvtColor(base_img0, cv2.COLOR_BGR2RGB)
        empty_img0 = copy(base_img0)
        empty_img0.fill(0)
        empty_img0_store = copy(empty_img0)
        empty_img0_store = np.int64(empty_img0_store)
        
        cursor.execute("SELECT COUNT(DISTINCT(frame)) \
                        FROM data \
                        WHERE camera_id = 0;")
        total_frames0 = cursor.fetchone()[0]
        cursor.execute("SELECT bbox_left, bbox_top, bbox_width, bbox_height, activity \
                        FROM data \
                        WHERE camera_id = 0;")
        for item in cursor.fetchall():
            if item[4] == 0:
                cv2.rectangle(empty_img0, (int(item[0]), int(item[1])), (int(item[2]),int(item[3])), (255,0,0), -1)
            elif item[4] == 1:
                cv2.rectangle(empty_img0, (int(item[0]), int(item[1])), (int(item[2]),int(item[3])), (0,0,255), -1)
            else:
                cv2.rectangle(empty_img0, (int(item[0]), int(item[1])), (int(item[2]),int(item[3])), (0,255,0), -1)
            empty_img0_store += empty_img0
            empty_img0.fill(0)
        
        empty_img0_store = empty_img0_store // total_frames0
        empty_img0_store = 255*empty_img0_store // (np.max(empty_img0_store))
        empty_img0_store = np.clip(empty_img0_store * 10, 0, 255)
        
        empty_img0_store = np.uint8(empty_img0_store)
        #empty_img_store = cv2.cvtColor(empty_img_store, cv2.COLOR_BGR2GRAY)
        #empty_img_store = cv2.cvtColor(empty_img_store, cv2.COLORMAP_JET)
        #empty_img_store = cv2.cvtColor(empty_img_store, cv2.COLOR_BGRA2BGR)
        heatmap_img0 = cv2.addWeighted(empty_img0_store, 0.5, base_img0, 0.5, 0)
        
        ax2.xaxis.set_visible(False)
        ax2.yaxis.set_visible(False)
        ax2.imshow(heatmap_img0)
        fig2.savefig('static/heatmap0.png')
        
        fig3, ax3 = plt.subplots()
        
        base_img1 = cv2.imread('static/camera1_empty.jpg',1)
        base_img1 = cv2.cvtColor(base_img1, cv2.COLOR_BGR2GRAY)
        base_img1 = cv2.cvtColor(base_img1, cv2.COLOR_BGR2RGB)
        empty_img1 = copy(base_img1)
        empty_img1.fill(0)
        empty_img1_store = copy(empty_img1)
        empty_img1_store = np.int64(empty_img1_store)
        
        cursor.execute("SELECT COUNT(DISTINCT(frame)) \
                        FROM data \
                        WHERE camera_id = 1;")
        total_frames1 = cursor.fetchone()[0]
        cursor.execute("SELECT bbox_left, bbox_top, bbox_width, bbox_height, activity \
                        FROM data \
                        WHERE camera_id = 1;")
        for item in cursor.fetchall():
            if item[4] == 0:
                cv2.rectangle(empty_img1, (int(item[0]), int(item[1])), (int(item[2]),int(item[3])), (255,0,0), -1)
            elif item[4] == 1:
                cv2.rectangle(empty_img1, (int(item[0]), int(item[1])), (int(item[2]),int(item[3])), (0,0,255), -1)
            else:
                cv2.rectangle(empty_img1, (int(item[0]), int(item[1])), (int(item[2]),int(item[3])), (0,255,0), -1)
            empty_img1_store += empty_img1
            empty_img1.fill(0)
        
        empty_img1_store = empty_img1_store // total_frames1
        empty_img1_store = 255*empty_img1_store // (np.max(empty_img1_store))
        empty_img1_store = np.clip(empty_img1_store * 10, 0, 255)
        
        empty_img1_store = np.uint8(empty_img1_store)
        #empty_img_store = cv2.cvtColor(empty_img_store, cv2.COLOR_BGR2GRAY)
        #empty_img_store = cv2.cvtColor(empty_img_store, cv2.COLORMAP_JET)
        #empty_img_store = cv2.cvtColor(empty_img_store, cv2.COLOR_BGRA2BGR)
        heatmap_img1 = cv2.addWeighted(empty_img1_store, 0.5, base_img1, 0.5, 0)
        
        ax2.xaxis.set_visible(False)
        ax2.yaxis.set_visible(False)
        ax2.imshow(heatmap_img1)
        fig2.savefig('static/heatmap1.png')
              
        
        if (connection.is_connected()):
            connection.commit()
            cursor.close()
            connection.close()
        else:
            return 'Error while connecting to MySQL'
    return render_template("dashboard.html", total_people = "Total customers: {}".format(total_people), 
                           total_people0 = "Total in camera 0: {}".format(total_people0),
                           total_people1 = "Total in camera 1: {}".format(total_people1),
                           total_people_sitting = "Total people who sat down: {}".format(total_people_sitting),
                           demovid = "{}.mp4".format(video_id),
                           plot = os.path.join('static','plot.png'),
                           customers_by_time = os.path.join('static','customers_by_time.png'),
                           heatmap0 = os.path.join('static','heatmap0.png'),
                           heatmap1 = os.path.join('static','heatmap1.png'))
    #return 'Total customers: {}'.format(row)
    #fig = create_figure()
    #FigureCanvas(fig).print_png(output)

#Define helper function for finding image paths
@app.route("/upload/<filename>")
def send_image(filename):
	return send_from_directory("uploads", filename)

#Start the application

if __name__ == "__main__":
	app.run(host='0.0.0.0', port=5000, debug=True)
