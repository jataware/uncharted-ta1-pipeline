FROM python:3.10-slim

RUN mkdir /app
RUN mkdir /workdir
WORKDIR /app

# install git and build tools for detectron2, ffmpeg  etc. for opencv
RUN apt-get update && apt-get install -y git build-essential ffmpeg libsm6 libxext6
RUN apt-get install -y libgdal-dev g++ --no-install-recommends && apt-get clean -y

ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# install torch separately to avoid conflicts with detectron2
RUN pip install torch==2.0.1

# install the tasks library with segmentation extras
COPY tasks tasks
RUN pip install -e ./tasks[segmentation]

# install the segmentation pipeline
COPY schema schema
COPY pipelines pipelines
RUN pip install -e ./pipelines/segmentation

ENTRYPOINT ["python3", "-m", "pipelines.segmentation.run_server"]


