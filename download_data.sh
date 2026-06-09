#!/bin/bash
set -e

mkdir -p data/coco data

echo "==> COCO train2017 (~18 GB)..."
wget -c "https://images.cocodataset.org/zips/train2017.zip" -O data/train2017.zip
unzip -q data/train2017.zip -d data/coco/
rm data/train2017.zip

echo "==> DTD (~600 MB)..."
wget -c "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz" -O data/dtd.tar.gz
tar -xzf data/dtd.tar.gz -C data/
rm data/dtd.tar.gz

echo "Done."
