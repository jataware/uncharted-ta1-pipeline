SEGMENTATION_MODEL ?=
POINTS_MODEL ?=
DEV_TAG := test
TAG := latest
IMAGE_NAMES := lara-cdr lara-georef lara-point-extract lara-segmentation lara-metadata-extract lara-text-extract

build_segmentation:
	@echo "\n*** Building segmentation ***"
	@cd pipelines/segmentation/deploy && ./build.sh $(SEGMENTATION_MODEL)

build_metadata:
	@echo "\n*** Building metadata extraction...\n"
	@cd pipelines/metadata_extraction/deploy && ./build.sh $(SEGMENTATION_MODEL)

build_points:
	@echo "\n*** Building point extraction ***"
	@cd pipelines/point_extraction/deploy && ./build.sh $(POINTS_MODEL) $(SEGMENTATION_MODEL)

build_georef:
	@echo "\n*** Building georeferencing *** "
	@cd pipelines/geo_referencing/deploy && ./build.sh $(SEGMENTATION_MODEL)

build_text:
	@echo "\n*** Building text extraction *** "
	@cd pipelines/text_extraction/deploy && ./build.sh

build_cdr:
	@echo "\n*** Building CDR mediator ***"
	@cd cdr/deploy && ./build.sh

build: build_segmentation build_metadata build_points build_georef build_cdr

tag_dev:
	@echo "*** Tagging images with dev tag [$(DEV_TAG)] ***"
	@$(foreach name, $(IMAGE_NAMES), \
		echo $(name):$(DEV_TAG);\
		docker tag uncharted/$(name):$(TAG) uncharted/$(name):$(DEV_TAG);)

push_dev:
	@echo "*** Pushing images with dev tag [$(DEV_TAG)] ***"
	@$(foreach name, $(IMAGE_NAMES), \
		echo $(name):$(DEV_TAG);\
		docker push uncharted/$(name):$(DEV_TAG);)

push:
	@echo "*** Tagging images with dev tag [$(TAG)] ***"
	@$(foreach name, $(IMAGE_NAMES), \
		echo $(name):$(TAG);\
		docker push uncharted/$(name):$(TAG);)

run:
	@echo "*** Running docker-compose ***"
	@cd deploy && docker compose up

