FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
MAINTAINER simonp6@kaist.ac.kr

WORKDIR /usr/src/app

COPY requirements.txt .

RUN pip install -r requirements.txt