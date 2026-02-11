#!/bin/bash
# Install custom CUDA ops for GaussianFormer

# Function to ensure pointops __init__.py has the correct import
ensure_pointops_init() {
    local init_file="$1"
    local required_line="from pointops.functions.pointops import furthestsampling as farthest_point_sampling"

    if [ ! -f "$init_file" ] || ! grep -Fxq "$required_line" "$init_file"; then
        echo "$required_line" > "$init_file"
        echo "__init__.py created/updated"
    else
        echo "__init__.py already correct"
    fi
}

# Step 1: Install required packages
pip install setuptools==69.5.1 packaging

cd model/encoder/gaussian_encoder/ops && pip install -e . && cd -
cd model/head/localagg && pip install -e . && cd -
# for GaussianFormer-2
cd model/head/localagg_prob && pip install -e . && cd -
cd model/head/localagg_prob_fast && pip install -e . && cd -

# Install pointops if not already installed
if ! pip show pointops &> /dev/null; then
    echo "Installing pointops..."
    if [ ! -d "pointops" ]; then
        git clone https://github.com/xieyuser/pointops.git
    else
        echo "pointops directory already exists, skipping clone"
    fi
    cd pointops
    ensure_pointops_init "__init__.py"
    python setup.py install
    cd -
else
    echo "pointops already installed, checking __init__.py..."
    if [ -d "pointops" ]; then
        ensure_pointops_init "pointops/__init__.py"
    fi
fi
