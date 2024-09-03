# Fairness Service

## Installation Instructions (python 3.9.12)

- git clone https://github.com/Mrasinthe/Fairness.git
- cd Fairness

## Add files
 - Add files from the link to the 'data' folder
   https://tartuulikool-my.sharepoint.com/:f:/g/personal/marasing_ut_ee/EkWFwdVoOX1PqYkxcAewgNoBGtxDn275IC9-Dt7uJ6qZ6g?e=Kc83V5

## Build and run the server using Docker

- sudo docker build -t fairness .
- sudo docker run -p 8083:8083 fairness

## Access Swagger UI documentation at:

- http://localhost:8083/docs
- Endpoint for the service: '/explain_fairness/file'

## Debugging
- sudo docker ps

## Clean rebuild docker image
- sudo docker rmi -f "docker_image_id" 
- sudo docker system prune


