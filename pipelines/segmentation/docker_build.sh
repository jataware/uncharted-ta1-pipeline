#!/bin/bash

# Script for creating 'segmenter' docker images
# and pushing to docker repo

DOCKER_IMAGE_TASKS="docker.uncharted.software/lara/segmenter-tasks"
DOCKER_FILE_TASKS="Dockerfile_tasks"
TASKS_CONTEXT="../../tasks/segmentation/"

DOCKER_IMAGE="docker.uncharted.software/lara/segmenter"
DOCKER_FILE="Dockerfile"

SEGMENTER_VERSION_TAG="0.1"

echo ""
echo "Version tag is "${SEGMENTER_VERSION_TAG}
echo "Building segmenter-tasks -- "${DOCKER_IMAGE_TASKS}":"${SEGMENTER_VERSION_TAG}" from docker file: "${DOCKER_FILE_TASKS}
echo ""

# Build segmenter 'tasks' image first
docker buildx build --platform=linux/amd64 -t ${DOCKER_IMAGE_TASKS}:${SEGMENTER_VERSION_TAG} -f ${DOCKER_FILE_TASKS} ${TASKS_CONTEXT}

echo ""
echo "Building segmenter -- "${DOCKER_IMAGE}":"${SEGMENTER_VERSION_TAG}" from docker file: "${DOCKER_FILE}
echo ""

# Build segmenter 'pipeline' image second
docker buildx build --platform=linux/amd64 -t ${DOCKER_IMAGE}:${SEGMENTER_VERSION_TAG} -f ${DOCKER_FILE} .


echo ""
echo "docker_build.sh done"
echo ""
