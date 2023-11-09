# Dockerfile for georef PIPELINE

# NOTES:
# - Must be run with build context ./
# - Uses 'docker.uncharted.software/lara/georef-tasks:0.1' as a base image

FROM  docker.uncharted.software/lara/georef-tasks:0.1 

# Create needed directories for georef pipeline
ARG PIPELINES_DIR=/app/pipelines/geo_referencing/
RUN mkdir -p ${PIPELINES_DIR}
WORKDIR ${PIPELINES_DIR}

COPY ../requirements.txt ./

# install requirements
RUN pip install --upgrade pip --no-cache-dir
RUN pip install --no-cache-dir -r requirements.txt      

# Copy over georef pipeline code
COPY ../*.py ../*.yaml ../*.yml ./
#COPY .env ./

CMD ["bash"]
