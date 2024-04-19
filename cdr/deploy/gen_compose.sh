#!/bin/bash
jinja --data ~/dev/lara_config/vars.json --format json --output docker-compose.yml docker-compose.j2 
