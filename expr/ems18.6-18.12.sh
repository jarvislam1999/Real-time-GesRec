#!/bin/bash

for expr in {18.6,18.7,18.8,18.9,18.10,18.11,18.12}; do
python expr/prepare_ems$expr.py;
done

for expr in {18.6,18.7,18.8,18.9,18.10,18.11,18.12}; do
bash ./expr/ems$expr.sh;
done