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
    bash_command = 'python3 run_activities.py /bucket/{} {} {}'.format(path, video_id, camera_id)
    result = subprocess.run(bash_command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,shell=True)
    return result.stdout.decode('utf-8')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

