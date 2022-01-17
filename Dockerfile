FROM ubuntu:latest

RUN apt update -y
RUN apt install python3.8 -y
RUN apt install python3-pip -y

RUN mkdir ml-devops-1
WORKDIR ml-devops-1

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY setup.py setup.py
COPY src src
RUN pip install -e .

COPY data data
COPY eda eda
COPY logs logs
COPY modelling_artifacts models