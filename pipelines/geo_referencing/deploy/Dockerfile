FROM docker-hub.uncharted.software/python:3.10-slim

RUN mkdir /app
RUN mkdir /workdir
WORKDIR /app

# install git and build tools ffmpeg etc.for opencv
RUN apt-get update && apt-get install -y git build-essential ffmpeg libsm6 libxext6
RUN apt-get install -y libgdal-dev g++ --no-install-recommends && apt-get clean -y

ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# install torch separately to avoid conflicts with detectron2
RUN pip install torch==2.0.1

COPY tasks tasks
RUN pip install -e ./tasks[segmentation]

COPY schema schema
COPY pipelines pipelines
COPY util util
COPY data data

# install lara packages - including point detection extras
RUN pip install -e ./pipelines/geo_referencing

ENTRYPOINT ["python3", "-m", "pipelines.geo_referencing.run_server"]