#!/bin/bash

# Configuration
REPO_ID="jgen/xray-comparative-evaluation"
LOCAL_DIR="./weights"

echo "Downloading model weights from Hugging Face..."
echo "Repository: $REPO_ID"
echo "Local directory: $LOCAL_DIR"
echo ""

# Check if huggingface-cli is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo "Error: huggingface-cli is not installed"
    echo "Please install it first by running: pip install huggingface_hub"
    echo "For more information, visit: https://huggingface.co/docs/huggingface_hub/en/guides/cli"
    exit 1
fi

# Create local directory if it doesn't exist
mkdir -p "$LOCAL_DIR"

# Download using Hugging Face CLI
echo "Starting download..."
huggingface-cli download "$REPO_ID" --local-dir "$LOCAL_DIR" --local-dir-use-symlinks False

if [ $? -eq 0 ]; then
    echo "Weights downloaded successfully to $LOCAL_DIR"
else
    echo "Failed to download weights"
    exit 1
fi