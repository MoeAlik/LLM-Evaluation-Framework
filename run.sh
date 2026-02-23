#!/bin/bash

set -euxo pipefail

pip3 install -r requirements.txt
python3 main.py
