#!/bin/bash

# args: $1 - path to JSON file containing the Jinja template variable values - see `vars_example.json` for the 
# the file structure.

jinja --data $1 --format json --output docker-compose.yml docker-compose.j2
