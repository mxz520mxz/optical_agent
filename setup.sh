#!/usr/bin/env bash
# Quick setup for optical_agent
set -e

echo "=== Cloning DeepLens ==="
if [ -d "DeepLens" ]; then
    echo "DeepLens/ already exists, skipping clone."
else
    git clone https://github.com/vccimaging/DeepLens.git
fi

echo "=== Installing Python dependencies ==="
pip install -r requirements.txt

echo "=== Setup complete ==="
echo "Run: python agent.py --help"
