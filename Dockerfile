FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
MAINTAINER simonp6@kaist.ac.kr

WORKDIR /usr/src/app

COPY requirements.txt .

RUN pip install -r requirements.txt