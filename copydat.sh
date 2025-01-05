#!/bin/bash

# Set base directories
SOURCE_DIR="../../new7/LightHART/data/"
DEST_DIR="data/smartfallmm"

# Create directory structure
mkdir -p $DEST_DIR/{old,young}/{accelerometer/{phone,watch},gyroscope/{phone,watch},skeleton}

# Function to copy files for each age group
copy_age_group() {
    AGE_GROUP=$1
    
    # Copy accelerometer data
    cp -r $SOURCE_DIR/$AGE_GROUP/accelerometer/phone/* $DEST_DIR/$AGE_GROUP/accelerometer/phone/
    cp -r $SOURCE_DIR/$AGE_GROUP/accelerometer/watch/* $DEST_DIR/$AGE_GROUP/accelerometer/watch/
    
    # Copy gyroscope data
    cp -r $SOURCE_DIR/$AGE_GROUP/gyroscope/phone/* $DEST_DIR/$AGE_GROUP/gyroscope/phone/
    cp -r $SOURCE_DIR/$AGE_GROUP/gyroscope/watch/* $DEST_DIR/$AGE_GROUP/gyroscope/watch/
    
    # Copy skeleton data
    cp -r $SOURCE_DIR/$AGE_GROUP/skeleton/* $DEST_DIR/$AGE_GROUP/skeleton/
}

# Copy files for both age groups
copy_age_group "old"
copy_age_group "young"

echo "Files copied successfully"
