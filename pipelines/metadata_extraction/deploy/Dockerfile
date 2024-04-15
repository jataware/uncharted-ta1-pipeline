FROM python:3.10-slim

RUN mkdir /app
RUN mkdir /workdir
WORKDIR /app

# install git and build tools ffmpeg etc.for opencv
RUN apt-get update && apt-get install -y git build-essential ffmpeg libsm6 libxext6
RUN apt-get install -y libgdal-dev g++ --no-install-recommends && apt-get clean -y

ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# install everythign needed for the segmentation task
RUN pip install torch==2.0.1
COPY tasks tasks
RUN pip install -e ./tasks[segmentation]

# install the metadata extraction
COPY schema schema
COPY pipelines pipelines
RUN pip install -e ./pipelines/metadata_extraction

ENTRYPOINT ["python3", "-m", "pipelines.metadata_extraction.run_server"]