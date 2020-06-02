cd ui
# sudo rm -r uploads
# mkdir uploads

nvidia-docker run -d --rm --ipc=host -p 8082:5000 -v /home/kellinpelrine/ui/uploads:/bucket tracker_container:latest

nvidia-docker run -d --rm -p 8083:5000 -v /home/kellinpelrine/ui/uploads:/bucket activities_container:latest

python ui_app.py
