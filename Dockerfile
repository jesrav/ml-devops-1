FROM ubuntu:latest

RUN apt update -y
RUN apt install python3.8 -y
RUN apt install python3-pip -y

RUN mkdir ml-devops-1
WORKDIR ml-devops-1

COPY data data
COPY images images
COPY logs logs
COPY results results
COPY models models

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt