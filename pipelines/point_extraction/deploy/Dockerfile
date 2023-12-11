FROM python:3.9-slim

RUN mkdir /app
RUN mkdir /workdir
WORKDIR /app

# install git and build tools for detectron2, ffmpeg  etc. for opencv
RUN apt-get update && apt-get install -y git build-essential ffmpeg libsm6 libxext6

# detectron2 and torch are not pip installable and need to be handled separately
COPY schema schema
COPY pipelines pipelines
COPY tasks tasks

# install lara packages - including point detection extras
RUN pip install -e ./tasks[point]
RUN pip install -e ./pipelines/point_extraction

ENTRYPOINT ["python3", "-m", "pipelines.point_extraction.run_server"]


