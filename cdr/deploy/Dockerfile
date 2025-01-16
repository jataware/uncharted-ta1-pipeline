FROM python:3.10-slim

# install git and build tools ffmpeg etc.for opencv
RUN apt-get update && apt-get install -y git build-essential ffmpeg libsm6 libxext6
RUN apt-get install -y libgdal-dev g++ --no-install-recommends && apt-get clean -y

# setup DOI root cert
COPY certs/DOIRootCA2.crt /usr/local/share/ca-certificates
RUN chmod 644 /usr/local/share/ca-certificates/DOIRootCA2.crt && \
    update-ca-certificates
ENV PIP_CERT="/etc/ssl/certs/ca-certificates.crt" \
    SSL_CERT_FILE="/etc/ssl/certs/ca-certificates.crt" \
    CURL_CA_BUNDLE="/etc/ssl/certs/ca-certificates.crt" \
    REQUESTS_CA_BUNDLE="/etc/ssl/certs/ca-certificates.crt" \
    AWS_CA_BUNDLE="/etc/ssl/certs/ca-certificates.crt"

RUN mkdir /app

COPY tasks tasks
RUN pip install -e ./tasks

COPY schema schema
COPY util util
COPY cdr cdr
RUN pip install -e ./cdr


ENTRYPOINT ["python3", "-m", "cdr.server"]
