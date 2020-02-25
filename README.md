3 step algorithm for tracking people in video data, with intended application to retail stores. Work in progress.

Data available at https://drive.google.com/drive/folders/17UMvm9GxCx6C4PxC3ZXb5rDUdFhUXJRd?usp=sharing.
The raw data is from http://imagelab.ing.unimore.it/visor/3dpes.asp, and has been processed from individual frames into videos using OpenCV, with the lossless HuffYUV codec. The processing program, 'img_to_video.ipynb,' is included in the 'data' folder here.

The prototype combines 3 algorithms in a potentially novel way (further literature review needed): YOLOv3 from https://github.com/AlexeyAB/darknet, Deepsort object tracking from https://github.com/abhyantrika/nanonets_object_tracking, and person re-identification from https://github.com/layumi/Person_reID_baseline_pytorch. The combination facilitates tracking between multiple cameras and fixes a common failure seen with pure tracking algorithms, person ID swapping (when two people move too near each other).

In progress: further testing, speed optimization and scaling, basic post-tracking analytics, full deployment pipeline.

Future research: societal effects, management of related personal data, potential associated technologies. 
