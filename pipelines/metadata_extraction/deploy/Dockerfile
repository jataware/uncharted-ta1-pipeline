FROM python:3.9-slim

RUN mkdir /app
RUN mkdir /input
RUN mkdir /output
RUN mkdir /workdir
WORKDIR /app

COPY schema schema
COPY pipelines pipelines
COPY tasks tasks

RUN pip install -e ./tasks
RUN pip install -e ./pipelines/metadata_extraction

ENV GOOGLE_APPLICATION_CREDENTIALS=/credentials.json

CMD ["python3", "-m", "pipelines.metadata_extraction.run_pipeline", "--input", "/input", "--output", "/output", "--workdir", "/workdir"]
