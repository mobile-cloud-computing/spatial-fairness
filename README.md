# Fairness Service

To assess whether a model is predicting the desirable outcome equally well for all values of a sensitive attribute.

## Prerequisites

- Python 3.9.12
- Docker installed if you wish to containerize the application

## Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/mobile-cloud-computing/spatial-fairness.git
```

Create and activate environment Variable:

```bash
python3 -m venv venv
source venv/bin/activate
```

Change directory to the cloned repository:

```bash
cd Fairness
```
Add files
Add files from the link to the 'data' folder https://tartuulikool-my.sharepoint.com/:f:/g/personal/marasing_ut_ee/EkWFwdVoOX1PqYkxcAewgNoBGtxDn275IC9-Dt7uJ6qZ6g?e=Kc83V5

Install python requirements:

```bash
pip install -r requirements.txt
```

## Run the application locally

```bash
python3 main.py
```

## Build and run the server using Docker

```bash
sudo docker build -t fairness .
sudo docker run -p 8083:8083 fairness
```
## Access Swagger UI documentation at:

- http://localhost:8083/docs
- Endpoint for the service: '/explain_fairness/file'

## Debugging

To view the docker containers and images

```bash
sudo docker ps
sudo docker images
```

## Clean rebuild docker image

```bash
sudo docker rmi -f <docker_image_id>
sudo docker system prune
```



