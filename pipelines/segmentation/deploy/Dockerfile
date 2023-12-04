FROM python:3.9-slim

RUN mkdir /app
RUN mkdir /workdir
WORKDIR /app

# install git and build tools for detectron2, ffmpeg  etc. for opencv
RUN apt-get update && apt-get install -y git build-essential ffmpeg libsm6 libxext6

# detectron2 and torch are not pip installable and need to be handled separately
ARG GPU
RUN if [[-z "${GPU}" == "gpu"]]; then \
        echo "Installing torch with GPU support"; \
        pip install torch==1.10.0+cu111 torchvision==0.11.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html; \
    else \
        echo "Installing torch without GPU support"; \
        pip install torch==1.10.0+cpu torchvision==0.11.0+cpu torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html;\
    fi
RUN if [[-z "${GPU}" == "gpu" ]]; then \
        echo "Installing dectectron2 with GPU support"; \
        pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html; \
    else \
        echo "Installing detectron2 without GPU support"; \
        pip install ninja; \
        python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'; \
    fi

COPY schema schema
COPY pipelines pipelines
COPY tasks tasks

# install lara packages - including segmentation extras (timm, transformers)
RUN pip install -e ./tasks[segmentation]
RUN pip install -e ./pipelines/segmentation

ENTRYPOINT ["python3", "-m", "pipelines.segmentation.run_server"]


