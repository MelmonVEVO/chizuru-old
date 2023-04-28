# syntax=docker/dockerfile:1
FROM tensorflow/tensorflow:2.11.0-gpu
WORKDIR /chizuru

COPY . .

RUN apt-get update \
    && apt-get install -y software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python3.7
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["python3.7", "chizuru.py"]