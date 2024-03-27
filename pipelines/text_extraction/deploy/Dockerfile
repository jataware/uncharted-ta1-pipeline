FROM python:3.10-slim

RUN mkdir /app
RUN mkdir /workdir
WORKDIR /app

# install git and build tools ffmpeg etc.for opencv
RUN apt-get update && apt-get install -y git build-essential ffmpeg libsm6 libxext6
RUN apt-get install -y libgdal-dev g++ --no-install-recommends && apt-get clean -y

ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

COPY schema schema
COPY pipelines pipelines
COPY tasks tasks

# install lara packages - including point detection extras
RUN pip install -e ./tasks
RUN pip install -e ./pipelines/text_extraction

ENTRYPOINT ["python3", "-m", "pipelines.text_extraction.run_server"]