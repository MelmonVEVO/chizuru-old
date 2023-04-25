# syntax=docker/dockerfile:1
FROM tensorflow/tensorflow:latest-gpu-py3
WORKDIR /chizuru
SHELL ["bash"]

COPY . .

RUN lscpi | grep -i nvidia
RUN apt-get update && apt-get install python3.7
RUN python3.7 -m venv ./venv
RUN ./venv/bin/activate
RUN pip -r requirements.txt

CMD ["python"]