# syntax=docker/dockerfile:1
FROM tensorflow/tensorflow:2.11.0-gpu
WORKDIR /chizuru

COPY . .

RUN apt-get update \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update
RUN apt-get install -y python3.7 python3.7-distutils
RUN python3.7 -m pip install --upgrade pip
RUN python3.7 -m pip install -r requirements.txt

CMD ["python3.7", "chizuru.py"]