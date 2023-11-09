FROM python:3.9.18-alpine3.18

RUN mkdir /app
RUN mkdir /input
RUN mkdir /output
RUN mkdir /workdir
WORKDIR /app

COPY pipelines pipelines
COPY tasks tasks

RUN pip install -e ./tasks
RUN pip install -e ./pipelines/geo_referencing

ENV GOOGLE_APPLICATION_CREDENTIALS=/credentials.json

CMD ["python3", "-m", "pipelines.geo_referencing.run_server"]