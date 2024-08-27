#!/bin/bash

# args: $1 - path to JSON file containing the Jinja template variable values - see `vars_example.json` for the 
# the file structure.

jinja2 --format json --outfile docker-compose.yml docker-compose.j2 $1 
