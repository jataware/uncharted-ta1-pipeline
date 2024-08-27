# LARA Stack Deployment

## Pre-requisites
1. A Python 3.10 environment
1. An activated NGROK account and token (see (https://ngrok.com/docs/getting-started/)[https://ngrok.com/docs/getting-started/])
1. An OpenAI API key
1. A Google Cloud Vision API JSON key *file*
1. A CriticalMAAS CDR API key (contact Jataware)

## Setup

Install Jinja-2 CLI:
```
pip install jinja2-cli
```

Make a copy of the `vars_example.json` file found in this directory and update it with the information specific to your environment.  

```
cp vars_example.json deploy_vars.json
```

The fields are:

* `work_dir`: A directory on the deployment host system that will store intermediate pipeline outputs 
* `image_dir`:  A directory on the deployment host system that will store COGs fetched from the CDR
* `cdr_api_token`: A CDR API token string provided by Jataware
* `ngrok_authtoken`:  NGROK auth token string 
* `openai_api_key`: Open AI API key string
* `google_application_credentials`: The path to the Google Cloud Vision API JSON key file   
* `tag`: The docker tag of the LARA images to deploy (ie. `latest`)

Generate a docker compose file from your variables:
```
./gen_compose deploy_vars.json
```

This should create a new `docker-compose.yml` files with values dervied from the `deploy_var.json` file.  LARA containers can now be pulled by running:
```
docker compose pull
```

The system can be started by running:
```
docker compose up
```

and stopped by running:
```
docker compose stop```


Once running, the system will respond to maps by being added to the CDR by executing the LARA pipelines and uploading results.