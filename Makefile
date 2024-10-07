# The path of the segmentation model to use
SEGMENTATION_MODEL ?=

# The path of the points model to use
POINTS_MODEL ?=

# command seperated list of platforms to build images for - example: linux/amd64,linux/arm64
# leaving unset will build for the host platform
PLATFORMS ?=

# Tag used for development images
DEV_TAG := test

# Baseline tag for stable images
TAG := latest

# List of images to build
IMAGE_NAMES := lara-cdr lara-cdr-writer lara-georef lara-point-extract lara-segmentation lara-metadata-extract lara-text-extract

# This Makefile contains build and deployment commands for various components of the LARA models project.

# Target: build_segmentation
# Description: Builds the segmentation component.
build_segmentation:
	@echo "*** Building segmentation ***\n"
	@cd pipelines/segmentation/deploy && ./build.sh $(SEGMENTATION_MODEL) $(PLATFORMS)

# Target: build_metadata
# Description: Builds the metadata extraction component.
build_metadata:
	@echo "*** Building metadata extraction ***\n"
	@cd pipelines/metadata_extraction/deploy && ./build.sh $(SEGMENTATION_MODEL) $(PLATFORMS)

# Target: build_points
# Description: Builds the point extraction component.
build_points:
	@echo "*** Building point extraction ***\n"
	@cd pipelines/point_extraction/deploy && ./build.sh $(POINTS_MODEL) $(SEGMENTATION_MODEL) $(PLATFORMS)

# Target: build_georef
# Description: Builds the georeferencing component.
build_georef:
	@echo "*** Building georeferencing ***\n"
	@cd pipelines/geo_referencing/deploy && ./build.sh $(SEGMENTATION_MODEL) $(PLATFORMS)

# Target: build_text
# Description: Builds the text extraction component.
build_text:
	@echo "*** Building text extraction ***\n"
	@cd pipelines/text_extraction/deploy && ./build.sh $(PLATFORMS)

# Target: build_cdr
# Description: Builds the CDR mediator component.
build_cdr:
	@echo "*** Building CDR mediator ***"\n
	@cd cdr/deploy && ./build.sh $(PLATFORMS)

# Target: build_cdr_writer
# Description: Builds the CDR writer component.
build_cdr_writer:
	@echo "*** Building CDR writer ***"\n
	@cd cdr_writer/deploy && ./build.sh $(PLATFORMS)


# Target: build
# Description: Builds all components.
build: build_segmentation build_metadata build_points build_georef build_text build_cdr build_cdr_writer

# Target: tag_dev
# Description: Tags images with the dev tag.
tag_dev:
	@echo "*** Tagging images with dev tag [$(DEV_TAG)] ***"
	@$(foreach name, $(IMAGE_NAMES), \
		echo $(name):$(DEV_TAG);\
		docker tag uncharted/$(name):$(TAG) uncharted/$(name):$(DEV_TAG);)

# Target: push_dev
# Description: Pushes images with the dev tag.
push_dev:
	@echo "*** Pushing images with dev tag [$(DEV_TAG)] ***"
	@$(foreach name, $(IMAGE_NAMES), \
		echo $(name):$(DEV_TAG);\
		docker push uncharted/$(name):$(DEV_TAG);)

# Target: push
# Description: Pushes images with the baseline tag.
push:
	@echo "*** Tagging images with dev tag [$(TAG)] ***"
	@$(foreach name, $(IMAGE_NAMES), \
		echo $(name):$(TAG);\
		docker push uncharted/$(name):$(TAG);)

# Target: run
# Description: Runs docker-compose.
run:
	@echo "*** Running docker-compose ***"
	@cd deploy && docker compose up
