#!/bin/bash
echo "Waiting for build to complete..."
for i in $(seq 1 30); do
    sleep 20
    CREATED=$(docker images caffe-cpu:customer --format '{{.CreatedAt}}' 2>/dev/null)
    echo "Check $i: Image created at $CREATED"
    if echo "$CREATED" | grep -q "2026-07-27 18:"; then
        echo ""
        echo "=== New image built successfully! ==="
        docker images caffe-cpu:customer --format 'table {{.Repository}}\t{{.Tag}}\t{{.CreatedAt}}\t{{.Size}}'
        exit 0
    fi
done
echo "Timeout waiting for build"
docker images caffe-cpu:customer
