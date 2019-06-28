#!/bin/bash

for expr in {13..19}; do
python expr/prepare_ems18.$expr.py;
done