#!/bin/bash
# Install custom CUDA ops for GaussianFormer

# Step 1: Install required packages
pip install setuptools==69.5.1 packaging

cd model/encoder/gaussian_encoder/ops && pip install -e . && cd -
cd model/head/localagg && pip install -e . && cd -
# for GaussianFormer-2
cd model/head/localagg_prob && pip install -e . && cd -
cd model/head/localagg_prob_fast && pip install -e . && cd -
