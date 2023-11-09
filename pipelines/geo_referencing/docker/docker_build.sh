#!/bin/bash

# Script for creating 'segmenter' docker images
# and pushing to docker repo

DOCKER_IMAGE_TASKS="docker.uncharted.software/lara/georef-tasks"
DOCKER_FILE_TASKS="Dockerfile_tasks"
TASKS_CONTEXT="../../../tasks/geo_referencing/"

DOCKER_IMAGE="docker.uncharted.software/lara/georef"
DOCKER_FILE="Dockerfile"

GEOREF_VERSION_TAG="0.1"

echo ""
echo "Version tag is "${GEOREF_VERSION_TAG}
echo "Building georef-tasks -- "${DOCKER_IMAGE_TASKS}":"${GEOREF_VERSION_TAG}" from docker file: "${DOCKER_FILE_TASKS}
echo ""

# Build georef 'tasks' image first
docker buildx build --platform=linux/amd64 -t ${DOCKER_IMAGE_TASKS}:${GEOREF_VERSION_TAG} -f ${DOCKER_FILE_TASKS} ${TASKS_CONTEXT}

echo ""
echo "Building georef -- "${DOCKER_IMAGE}":"${GEOREF_VERSION_TAG}" from docker file: "${DOCKER_FILE}
echo ""

# Build georef 'pipeline' image second
docker buildx build --no-cache --platform=linux/amd64 -t ${DOCKER_IMAGE}:${GEOREF_VERSION_TAG} -f ${DOCKER_FILE} .


echo ""
echo "docker_build.sh done"
echo ""
