# Use an official Python runtime as a parent image
FROM python:3.9.12

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
# COPY . /app
# COPY model /app/model

# Install any needed packages specified in requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

# Make port 8083 available to the world outside this container
EXPOSE 8083

# Define environment variable
# ENV NAME World

# Run app.py when the container launches
CMD ["python", "main.py"]

########### Old ###########
# Start from a minimal NVIDIA CUDA base image
#FROM nvidia/cuda:12.2.0-base-ubuntu22.04

# Install Python 3.9 and other necessary packages
#RUN apt-get update && apt-get install -y python3.10 python3-tk && \
#    ln -s /usr/bin/python3.10 /usr/local/bin/python3 && \
#    apt-get clean && \
#    rm -rf /var/lib/apt/lists/*

# Install pip
#RUN apt-get update && \
#    apt-get install -y curl && \
#    curl https://bootstrap.pypa.io/get-pip.py | python3 && \
#    apt-get remove -y curl && \
#    apt-get clean && \
#    rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
#WORKDIR /app

# Copy only the necessary files
#COPY requirements.txt .
#COPY . .

# Install any needed packages specified in requirements.txt
#RUN pip3 install --no-cache-dir -r requirements.txt

# Make port 8083 available to the world outside this container
#EXPOSE 8083

# Run main.py when the container launches
#CMD ["python3", "main.py"]
