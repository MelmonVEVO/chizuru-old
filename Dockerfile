# syntax=docker/dockerfile:1
FROM tensorflow/tensorflow:2.11.0-gpu
WORKDIR /chizuru
SHELL ["bash"]

COPY . .

RUN apt update && apt install -y software-properties-common && add-apt-repository -y ppa:deadsnakes/ppa && apt-get update && apt install -y python3.7
RUN python3.7 -m venv ./venv
RUN ./venv/bin/activate
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD python3.7 chizuru.py