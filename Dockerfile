# syntax=docker/dockerfile:1
FROM tensorflow/tensorflow:latest-gpu-py3
WORKDIR /chizuru
SHELL ["bash"]

COPY . .
RUN lscpi | grep -i nvidia
RUN pip3 -r requirements.txt
RUN python3 train.py