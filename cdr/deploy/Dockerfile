FROM python:3.10-slim

RUN mkdir /app

COPY cdr cdr
COPY schema schema
COPY tasks tasks
COPY process process

RUN pip install -e ./tasks
RUN pip install -e ./cdr

ENTRYPOINT ["python3", "-m", "cdr.server"]
