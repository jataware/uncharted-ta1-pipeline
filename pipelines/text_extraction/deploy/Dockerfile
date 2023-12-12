FROM python:3.9-slim

RUN mkdir /app
RUN mkdir /workdir
WORKDIR /app

# install git and build tools ffmpeg etc.for opencv
RUN apt-get update && apt-get install -y git build-essential ffmpeg libsm6 libxext6

COPY schema schema
COPY pipelines pipelines
COPY tasks tasks

# install lara packages - including point detection extras
RUN pip install -e ./tasks
RUN pip install -e ./pipelines/text_extraction

ENTRYPOINT ["python3", "-m", "pipelines.text_extraction.run_server"]