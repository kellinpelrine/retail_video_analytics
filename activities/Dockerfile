FROM nvidia/cuda:latest

MAINTAINER Kellin Pelrine "kellinpelrine@gmail.com"


RUN apt-get update -y && \
    apt-get install -y python3-pip python3-dev git wget && \
    pip3 install --upgrade pip

RUN apt-get install -y libsm6 libxext6 libxrender-dev curl

RUN pip3 install opencv-python pandas mysql-connector-python

COPY ./run_activities.py /app/run_activities.py

# We copy just the requirements.txt first to leverage Docker cache
COPY ./requirements.txt /app/requirements.txt

WORKDIR ../app

RUN pip3 install -r requirements.txt

COPY . /app

RUN apt update
RUN apt install ffmpeg -y

ENV PORT 8082

ENTRYPOINT [ "python3" ]

CMD [ "app.py" ]
