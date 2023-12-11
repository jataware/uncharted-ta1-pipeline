FROM python:3.9-slim

RUN mkdir /app
RUN mkdir /workdir
WORKDIR /app

# install git and build tools for detectron2, ffmpeg  etc. for opencv
RUN apt-get update && apt-get install -y git build-essential ffmpeg libsm6 libxext6

COPY schema schema
COPY pipelines pipelines
COPY tasks tasks

# install lara packages - including segmentation extras (timm, transformers)
RUN pip install -e ./tasks[segmentation]
RUN pip install -e ./pipelines/segmentation

ENTRYPOINT ["python3", "-m", "pipelines.segmentation.run_server"]


