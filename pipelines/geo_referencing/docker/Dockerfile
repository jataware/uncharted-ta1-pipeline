FROM docker-hub.uncharted.software/python:3.9-slim 

RUN mkdir /app
RUN mkdir /input
RUN mkdir /output
RUN mkdir /workdir
WORKDIR /app

COPY compute compute
COPY model model
COPY util util
COPY schema schema
COPY pipelines pipelines
COPY tasks tasks

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install -e ./tasks
RUN pip install -e ./pipelines/geo_referencing

ENV GOOGLE_APPLICATION_CREDENTIALS=/credentials.json

CMD ["python3", "-m", "pipelines.geo_referencing.run_server"]