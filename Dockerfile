# Use a base image with CUDA support
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y python3.10 python3-pip python3-venv

ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH

# Set working directory
WORKDIR /app

RUN python3 -m venv venv

# Copy requirements file
COPY requirements-gpu.txt .

# Install Python dependencies
RUN venv/bin/pip3 install --no-cache-dir -r requirements-gpu.txt

# Copy your application code
COPY *.py .

# Expose the port your Quart app runs on
EXPOSE 5001

# Command to run your Quart app
CMD ["venv/bin/quart", "run"]
