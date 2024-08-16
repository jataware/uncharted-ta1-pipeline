# The path of the segmentation model to use
SEGMENTATION_MODEL ?=

# The path of the points model to use
POINTS_MODEL ?=

# Tag used for development images
DEV_TAG := test

# Baseline tag for stable images
TAG := latest

# List of images to build
IMAGE_NAMES := lara-cdr lara-georef lara-point-extract lara-segmentation lara-metadata-extract lara-text-extract

# This Makefile contains build and deployment commands for various components of the LARA models project.

# Target: build_segmentation
# Description: Builds the segmentation component.
build_segmentation:
	@echo "\n*** Building segmentation ***"
	@cd pipelines/segmentation/deploy && ./build.sh $(SEGMENTATION_MODEL)

# Target: build_metadata
# Description: Builds the metadata extraction component.
build_metadata:
	@echo "\n*** Building metadata extraction...\n"
	@cd pipelines/metadata_extraction/deploy && ./build.sh $(SEGMENTATION_MODEL)

# Target: build_points
# Description: Builds the point extraction component.
build_points:
	@echo "\n*** Building point extraction ***"
	@cd pipelines/point_extraction/deploy && ./build.sh $(POINTS_MODEL) $(SEGMENTATION_MODEL)

# Target: build_georef
# Description: Builds the georeferencing component.
build_georef:
	@echo "\n*** Building georeferencing *** "
	@cd pipelines/geo_referencing/deploy && ./build.sh $(SEGMENTATION_MODEL)

# Target: build_text
# Description: Builds the text extraction component.
build_text:
	@echo "\n*** Building text extraction *** "
	@cd pipelines/text_extraction/deploy && ./build.sh

# Target: build_cdr
# Description: Builds the CDR mediator component.
build_cdr:
	@echo "\n*** Building CDR mediator ***"
	@cd cdr/deploy && ./build.sh

# Target: build
# Description: Builds all components.
build: build_segmentation build_metadata build_points build_georef build_text build_cdr

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
