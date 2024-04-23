#!/bin/bash

# args: $1 - path to JSON file containing the Jinja template variable values

jinja --data $1 --format json --output docker-compose.yml docker-compose.j2
