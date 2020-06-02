FROM nvidia/cuda:latest

MAINTAINER Kellin Pelrine "kellinpelrine@gmail.com"


RUN apt-get update -y && \
    apt-get install -y python3-pip python3-dev git wget && \
    pip3 install --upgrade pip


RUN git clone https://github.com/AlexeyAB/darknet && \
    git clone https://github.com/abhyantrika/nanonets_object_tracking && \
    git clone https://github.com/layumi/Person_reID_baseline_pytorch.git && \
    cp -r Person_reID_baseline_pytorch/* nanonets_object_tracking/

RUN apt-get install -y libsm6 libxext6 libxrender-dev curl

WORKDIR /nanonets_object_tracking/
RUN pip3 install -r requirements.txt && \
    pip3 install pretrainedmodels && \
    pip3 install pyyaml && \
    mkdir './model/PCB' && \
    wget 'https://docs.google.com/uc?export=download&id=1dyoJ04vMowMkRt5QHiEF1dn2bstlxrw4' -O './model/PCB/opts.yaml' && \
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=10_l0oSL4Cjae_quLw9lStOanKpSr5WQy' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=10_l0oSL4Cjae_quLw9lStOanKpSr5WQy" -O './model/PCB/net_last.pth' && rm -rf /tmp/cookies.txt

WORKDIR ../darknet
RUN sed -i 's/GPU=0/GPU=1/g' Makefile && \
    make && \
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=15QbBtXf_A5X26oOwbIenAzntqAKnHb6m' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=15QbBtXf_A5X26oOwbIenAzntqAKnHb6m" -O 'yolov3.weights' && rm -rf /tmp/cookies.txt

COPY ./run_tracker_reid.py /nanonets_object_tracking/run_tracker_reid.py

# We copy just the requirements.txt first to leverage Docker cache
COPY ./requirements.txt /app/requirements.txt

WORKDIR ../app

RUN pip3 install -r requirements.txt

COPY . /app

ENV PORT 8082

ENTRYPOINT [ "python3" ]

CMD [ "app.py" ]
