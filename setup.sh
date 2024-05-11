#!/bin/bash

# Define directories
RAW_DIR="dev_data/raw"
PROCESSED_DIR="dev_data/processed"

# Create directories if they don't exist
mkdir -p "$RAW_DIR"
mkdir -p "$PROCESSED_DIR"

# Download dataset
curl -L "https://zenodo.org/records/7882613/files/dev_gearbox.zip?download=1" -o "$RAW_DIR/gearbox.zip"

# Unzip dataset
unzip "$RAW_DIR/gearbox.zip" -d "$RAW_DIR"

# Remove the downloaded zip file
rm "$RAW_DIR/gearbox.zip"

# Move train data to appropriate directory
mv "$RAW_DIR/gearbox/train" "$RAW_DIR/gearbox/normal"