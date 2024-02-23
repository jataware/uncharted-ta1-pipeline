FROM python:3.9-slim

RUN mkdir /app
RUN mkdir /workdir
WORKDIR /app

# install git and build tools ffmpeg etc.for opencv
RUN apt-get update && apt-get install -y git build-essential ffmpeg libsm6 libxext6

COPY schema schema
COPY pipelines pipelines
COPY tasks tasks
COPY util util

# install lara packages - including point detection extras
RUN pip install torch==2.0.1
RUN pip install -e ./tasks[segmentation]
RUN pip install -e ./pipelines/geo_referencing

ENTRYPOINT ["python3", "-m", "pipelines.geo_referencing.run_server"]