# Fairness Service

## Installation Instructions (python 3.8.10)

- git clone https://github.com/Mrasinthe/Fairness.git
- cd Fairness

## Build and run the server

- sudo docker build -t fairness .
- sudo docker run -p 8083:8083 fairness

# Debugging
sudo docker ps

# Clean rebuild docker image
sudo docker rmi -f "docker_image_id" 
sudo docker system prune


