#!/bin/bash
jinja --data $1 --format json --output docker-compose.yml docker-compose.j2 
