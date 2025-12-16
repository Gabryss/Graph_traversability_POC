#!/bin/bash
#xhost +local:docker

# Note: replace --restart=always to --rm for testing
docker run -it \
    --rm \
    --network=host \
    --ipc=host \
    --pid=host \
    --privileged \
    --env DISPLAY=$DISPLAY \
    --volume /tmp/.X11-unix:/tmp/.X11-unix \
    --memory="2048m"\
    --env UID=$(id -u) \
    --env GID=$(id -g) \
    --volume="${PWD}/../src:/root/src" \
    --volume="${PWD}/../data:/root/data" \
    --volume="${PWD}/../config:/root/config" \
    --workdir /root \
    --name graph_traversability \
    local/graph_traversability:poc \
    python3 src/run_pipeline.py
