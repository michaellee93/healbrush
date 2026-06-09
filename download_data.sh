#!/bin/bash
# Pull training data from R2. Requires AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY set.
set -e

R2_ENDPOINT="https://4da15aa084cf06c245300b38621c9cfe.r2.cloudflarestorage.com"
BUCKET="healbrush-data"

mkdir -p data/coco/train2017 data/dtd

echo "==> COCO train2017..."
aws s3 sync "s3://${BUCKET}/coco/train2017/" data/coco/train2017/ \
    --endpoint-url "$R2_ENDPOINT" --no-progress

echo "==> DTD..."
aws s3 sync "s3://${BUCKET}/dtd/" data/dtd/ \
    --endpoint-url "$R2_ENDPOINT" --no-progress

echo "Done."
