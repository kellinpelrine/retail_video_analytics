# flask_web/app.py

from flask import Flask, request
import subprocess
import os
app = Flask(__name__)

@app.route('/run')
def hello_world():
    path = request.args.get('path')
    video_id = request.args.get('video_id')
    camera_id = request.args.get('camera_id')
    os.chdir('/app')
    bash_command = 'python3 run_yolo.py /bucket/:{}'.format(path)
    result = subprocess.run(bash_command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,shell=True)
    os.chdir('/nanonets_object_tracking/')
    bash_command = 'python3 run_tracker_reid.py --path /bucket/{} --video_id {}'.format(path, video_id)
    result = subprocess.run(bash_command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,shell=True)

    # call activities
    bash_command = 'curl \"34.74.59.41:8083/run?video_id={}&camera_id={}&path={}\"'.format(video_id, camera_id, path)
    result = subprocess.run(bash_command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,shell=True)
    return result.stdout.decode('utf-8')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

