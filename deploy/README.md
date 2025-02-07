# LARA Stack Deployment

![LARA Architecture](../images/lara_deployment.png)

## Pre-requisites
1. A Python 3.10 environment
1. An activated NGROK account and token (see https://ngrok.com/docs/getting-started/)
1. An OpenAI API key or Azure OpenAI API key and endpoint
1. A Google Cloud Vision API JSON key *file* (see https://cloud.google.com/vision/docs/ocr)
1. A CriticalMAAS CDR API key (contact Jataware)
1. A writeable S3 bucket, and its associated credentials

## Setup

Install Jinja-2 CLI:
```console
pip install jinja2-cli
```

Make a copy of the `vars_example.json` file found in this directory and update it with the information specific to your environment.

```console
cp vars_example.json deploy_vars.json
```

The fields that will need to be set are:

* `work_dir`: S3 bucket URL where intermediate pipeline outputs will be cached
* `image_dir`: S3 bucket URL where COGs fetched from the CDR will be cached
* `aws_access_key_id`: The AWS access key ID for the `workdir` anad `imagedir` S3 bucket
* `aws_secret_access_key`: The AWS secret access key for the `workdir` anad `imagedir` S3 bucket
* `cdr_api_token`: A CDR API token string provided by Jataware
* `cdr_host`: URL of the CDR host
* `cog_host`: URL of the COG storage server
* `ngrok_authtoken`: NGROK auth token string - used when NGROK will provide the CDR webhook URL.
* `cdr_callback_url`: The URL to use for CDR webhook access - should be the `cdr` container URL, and be reachable from the CDR itself.  Not used if an NGROK auth token is supplied.
* `llm_provider`:  set to `azure` or `openai` based on the desired LLM host - defaults to `openai`
* `llm`: the model identification string - defaults to `gpt-4o`, valid options are determined by the model provider
* `azure_openai_api_key`: Azure OpenAI API key string - used when `llm_provider` is set to `azure`
* `azure_openai_endpoint`: The endpoint of the Azure OpenAI service - used when `llm_provider` is set to `azure`
* `openai_api_key`: OpenAI API key for OpenAI's hosted service (only needed if not using Azure) - used when `llm_provider` is set to `openai` (default)
* `google_application_credentials_dir`: A path pointing to the **directory** containing the Google Cloud Vision API JSON key **file**. ⚠️ **NOTE:** This file MUST be named `google_application_credentials.json`.⚠️ Exclude this argument if USGS cloud auth is used.
* `tag`: The docker tag of the LARA images to deploy (ie. `latest`)
* `gpu`: A boolean indicating whether or not the system should attempt to use GPU resources if available.

Next, generate a docker compose file from your variables:
```console
./gen_compose.sh deploy_vars.json
```

This should create a new `docker-compose.yml` file with values derived from the `deploy_var.json` file.  The compose file can be copied to an intended deployment environment such as an AWS EC2 node.

LARA containers can be pulled by running:
```console
docker compose pull
```

The system can be started in detached mode by running:
```console
docker compose up -d
```

and stopped by running:
```console
docker compose stop
```

Status can be checked using:
```console
docker compose ps
```

Logs are available through:
```console
docker compose logs -f
```

Once running, the system will respond to maps by being added to the CDR by executing the LARA pipelines and uploading results.  This can be triggered through the CriticalMAAS Polymer HMI by selecting a map and clicking the "Process Map" button.  Logs will display processing progress.

## Deployment Recommendations

The system has been tested on an AWS `m6i.2xlarge node` (32GB RAM, 8x vCPU, NO GPU), which results in processing times of 2-3 minutes per individual map.  A node with 16GB of RAM was tested, but larger maps failed to process due to out-of-memory errors.

The S3 bucket used for the `workdir` and `imagedir` storage should have an expiry period set (24 hrs. recommended).  The data stored during processing is not needed after results are written to the CDR, but the application **does not** clean up these files itself.

## Monitoring and Troubleshooting

### CDR Requests

Map processing requests initiated by Polymer users are managed by the LARA CDR mediator, which runs as the `cdr` service.  For each request, logs will be generated to indicated that the request has been received, and is being passed to the model services for processing. To view the cdr container logs:

```console
docker compose logs -f cdr
```

Logs in the example below show a COG with the ID `e0022ba723b114b236b298fa9cf535652a614e614c655d2553bb43f35e331e70` being processed by the segmentation and metadata models:

```
2025-02-06 14:45:29 cdr-799d4bbb49-vfpqs chaining_result_subscriber[1] INFO received data from result channel
2025-02-06 14:45:29 cdr-799d4bbb49-vfpqs chaining_result_subscriber[1] INFO processing result for request segmentation-1738759133-pipeline of type 3
2025-02-06 14:45:29 cdr-799d4bbb49-vfpqs chaining_result_subscriber[1] INFO sending write request: segmentation
2025-02-06 14:45:29 cdr-799d4bbb49-vfpqs request_publisher[1] INFO sending request segmentation-1738759133-pipeline for image e0022ba723b114b236b298fa9cf535652a614e614c655d2553bb43f35e331e70 to lara queue
2025-02-06 14:45:29 cdr-799d4bbb49-vfpqs request_publisher[1] INFO request segmentation-1738759133-pipeline published to write_request
2025-02-06 14:45:29 cdr-799d4bbb49-vfpqs chaining_result_subscriber[1] INFO sending next request in sequence: metadata
2025-02-06 14:45:29 cdr-799d4bbb49-vfpqs request_publisher[1] INFO sending request metadata-1738853129-pipeline for image e0022ba723b114b236b298fa9cf535652a614e614c655d2553bb43f35e331e70 to lara queue
2025-02-06 14:45:29 cdr-799d4bbb49-vfpqs request_publisher[1] INFO request metadata-1738853129-pipeline published to metadata_request
2025-02-06 14:45:29 cdr-799d4bbb49-vfpqs chaining_result_subscriber[1] INFO result processing finished
```

If users initiate a map processing request and corresponding logs are not generated, the most likely cause will be a mis-registration of the LARA CDR mediator.  Restarting the `cdr` service will force the system to re-register with the CDR and will likely resolve the issue.

### CDR Result Uploads

Map extractions are stored as JSON data which is uploaded to the CDR by the `cdr_writer` service.  In the case of georeferencing, a GeoTiff projected into WGS84 is also produced an uploaded.  Each of the 4 map processing services generates output that is viewable in the logs using the following command:

```console
docker compose logs -f cdr_writer
```

The following logs demonstrate the segmentation results for COG `d24108d6753773337b3733b75eb132b1153558b5487159755075517549f10370` being successfully uploaded to the CDR by the `cdr_writer`:

```
2025-02-06 15:02:59 writer-6684c8b5fb-49fhq write_result_subscriber[1] INFO received data from result channel
2025-02-06 15:02:59 writer-6684c8b5fb-49fhq write_result_subscriber[1] INFO processing result for request segmentation-1738759122-pipeline for d24108d6753773337b3733b75eb132b1153558b5487159755075517549f10370 of type 3
2025-02-06 15:02:59 writer-6684c8b5fb-49fhq write_result_subscriber[1] INFO segmentation results received
2025-02-06 15:02:59 writer-6684c8b5fb-49fhq write_result_subscriber[1] INFO pushing features result for request segmentation-1738759122-pipeline to CDR
2025-02-06 15:02:59 writer-6684c8b5fb-49fhq httpx[1] INFO HTTP Request: POST https://api.cdr.land/v1/maps/publish/features "HTTP/1.1 200 OK"
2025-02-06 15:02:59 writer-6684c8b5fb-49fhq write_result_subscriber[1] INFO result for request segmentation-1738759122-pipeline sent to CDR with response 200: b'{"id":"uncharted-area_1.0.0_d24108d6753773337b3733b75eb132b1153558b5487159755075517549f10370","elasticsearch_response":{"_index":"extraction_results","_id":"uncharted-area_1.0.0_d24108d6753773337b3733b75eb132b1153558b5487159755075517549f10370","_version":1,"result":"created","forced_refresh":true,"_shards":{"total":1,"successful":1,"failed":0},"_seq_no":179317,"_primary_term":1},"job_id":"19ba84c984e5495cbe362b59a22b5b7d"}'
2025-02-06 15:03:00 writer-6684c8b5fb-49fhq write_result_subscriber[1] INFO result processing finished
```

Failures to write to the CDR are generally a result of availablity or processing issues on the CDR itself; HTTP error codes and reasons will be present in the logs to aid in troubleshooting.

### Map Processing Services

Maps are processed sequentially by the four extraction services: `segmentation`, `metadata`, `points` and `georef`, each with their own specific logs:

```console
docker compose logs -f segmentation
docker compose logs -f metadata
docker compose logs -f points
docker compose logs -f georef
```

These services make use of the external APIs in the table below **which must be configured correctly in order for the containers to start**.  Logs specific to each container will indicate any configuration errors on startup.

| Service | OpenAI GPT | GoogleVision OCR |
| --------| ------------ | --------- |
| `segmentation` | :x: | :white_check_mark: |
| `metadata` | :white_check_mark: | :white_check_mark: |
| `points` | :x: | :white_check_mark: | :x: |
| `georef` | :white_check_mark: | :white_check_mark: |

Each service generates status logs specific to their processing, with the ID of the COG being processed logged as part of status reporting.  In some cases, errors may be encountered due to the nature of the input image, or due to availabilty issues with external APIs.  The system will attempt to retry a given map a number of times (default is 3) but will eventually discard it if errors continue to be encountered.  Logs capturing the errors will be present, with an indication of the source of the error (internal processing or external API) along with the ID of the map.

### RabbitMQ

The LARA stack uses RabbitMQ for queueing and managing processing tasks, with each map processing service, and the CDR writer having their own mesage queues.  The admin console is available on port 15672, and can be useful for tracking the progress of map processing, especially when larger numbers of jobs are being queued.  As configured, jobs will initially appear in the `segmentation_request_queue`, and move through `metadata_request_queue`, `point_request_queue`, `georef_request_queue`.  Each service's output will be added to the `write_request` queue when map processing is complete.
