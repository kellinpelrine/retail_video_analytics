3 step algorithm for tracking people in video data, with intended application to retail stores.

![](https://github.com/kellinpelrine/retail_video_analytics/blob/master/Dashboard%20video%201.gif)

The system here combines 3 algorithms: YOLOv3 from https://github.com/AlexeyAB/darknet, Deepsort object tracking from https://github.com/abhyantrika/nanonets_object_tracking, and person re-identification from https://github.com/layumi/Person_reID_baseline_pytorch. The combination facilitates tracking between multiple cameras and fixes a common failure seen with pure tracking algorithms, person ID swapping (when two people move too near each other).

Deployment is facilitated through docker containers, flask web servers, and a MySQL db. It is designed to be deployed using Google Cloud Platform, which also facilitates easy multiple deployment by saving the virtual machine image of the first deployment.

To deploy for the first time on Google cloud, do the following. In subsequent deployments, you can save the disk image and skip steps 4 and 6, as well as 2 and 5 if using the same database.

1. Create virtual machine with GPU.
2. Create MySQL database.
- create database named "output_data"
- run the following:
  CREATE TABLE data
  (
  idx SERIAL PRIMARY KEY,
  frame INTEGER NOT NULL,
  person_id INTEGER NOT NULL,
  bbox_left REAL NOT NULL,
  bbox_top REAL NOT NULL,
  bbox_width REAL NOT NULL,
  bbox_height REAL NOT NULL,
  activity INTEGER NOT NULL,
  video_id INTEGER NOT NULL,
  camera_id INTEGER NOT NULL
  );
3. Follow directions at https://cloud.google.com/sql/docs/mysql/connect-compute-engine#gce-connect-ip to enable connection between VM and database. Also on the vm allow access to port 5000 to access the dashboard and 8083 to call the activities container.
4. Upload ui, activities, and inference folders to VM.
5. Modify db_info.yaml in ui folder with the credentials to connect to your db.
6. cd into the appropriate directories (inference, activities respectively) and build containers with nvidia-docker: 
- nvidia-docker build -t tracker_container:latest .
- nvidia-docker build -t activities_container:latest .
7. Modify ui_app.py for your desired number of cameras. Make static directory under ui and upload base images from each camera for heatmaps, named "camera0_empty.jpg" or the appropriate camera IDs. These are for heatmaps on the dashboard (see below).
8. Upload startup.sh and run it with "bash startup.sh" to start the containers and web interface.
9. Go to virtual-machine-ip:5000 to access the dashboard.

From here, you can upload videos (multiple per upload if desired) with the name convention "camera0_video0" where the zeros are replaced with the desired camera and video numbers. Once processing is complete, the website shows how to view the results.

Dashboard shows information about the videos uploaded, including total number of people/customers, people in each camera, time each person is present, people/customers by time of day, anotated video including activities (moving = green, standing = blue, sitting = red, in this prototype), and heatmaps showing where people have been like these:
![image](https://github.com/kellinpelrine/retail_video_analytics/blob/master/Dashboard%20picture%201.png)

Main data available at https://drive.google.com/drive/folders/17UMvm9GxCx6C4PxC3ZXb5rDUdFhUXJRd?usp=sharing.
The raw data is from http://imagelab.ing.unimore.it/visor/3dpes.asp, and has been processed from individual frames into videos using OpenCV, with the lossless HuffYUV codec. The processing program, 'img_to_video.ipynb,' is included in the 'data' folder here.

Future research: modular framework to add and modify components of this, more components, societal effects, management of related personal data, potential associated technologies. 
