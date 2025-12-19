# Use the official ROS2 image as the base image
FROM python:3.10-slim
LABEL maintainer="Gabriel Garcia <gabriel.garcia@uni.lu>"


RUN apt-get update && \
    pip install --no-cache-dir perlin-noise networkx matplotlib numpy pykdtree scipy && \
    rm -rf /var/lib/apt/lists/*

#RUN apt update && \
#pip install noise networkx matplotlib numpy && \
#rm -rf /var/lib/apt/lists/*
# last line to remove unecessary stuff
# and reduce image size

# Set up entrypoint to source ROS2 setup files
COPY entrypoint.sh /
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

RUN mkdir -p /root/src/

# Set the default command to run when the container starts
CMD ["bash"]
