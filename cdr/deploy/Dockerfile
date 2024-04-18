FROM python:3.10-slim

# install git and build tools ffmpeg etc.for opencv
RUN apt-get update && apt-get install -y git build-essential ffmpeg libsm6 libxext6
RUN apt-get install -y libgdal-dev g++ --no-install-recommends && apt-get clean -y

RUN mkdir /app

COPY tasks tasks
RUN pip install -e ./tasks

COPY schema schema
COPY util util
COPY cdr cdr
RUN pip install -e ./cdr


ENTRYPOINT ["python3", "-m", "cdr.server"]
