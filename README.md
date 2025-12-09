# Alpha-Gamma Predictions
## using multiple models

## Conda
```
conda create -n SLCenv python=3.11 -y

conda activate SLCenv

pip install -r requirements
```

## DVC
```
git init

dvc init

dvc repro
```

## Flask
```
pip install flask

# start backend
python main.py

# change directory to frontend folder
cd flask_api/frontend

python -m http.server 8000

# visit
http://localhost:8000


```

## MLFLOW on AWS

- create IAM user (AdminstratorAccess), EC2 instance (Ubuntu with security groups 5000 port) and S3 bucket

- In local console

` aws configure 

- In EC2 Machine
```
sudo apt update

sudo apt install python3-pip

sudo apt install pipenv

sudo apt install virtualenv

mkdir mlflow

cd mlflow

pipenv install mlflow

pipenv install awscli

pipenv install boto3

pipenv shell

mlflow server -h 0.0.0.0 --default-artifact-root s3://stanley-mlflow-bucket-27 --allowed-hosts "ec2-13-244-77-114.af-south-1.compute.amazonaws.com:5000"
```

- set URI in local terminal or code

mlflow_tracking_uri:

`http://ec2-13-244-77-114.af-south-1.compute.amazonaws.com:5000

## DOCKER
```
docker build -t mcstanleydocker27/youtubeinsights:latest .

docker run -p 5000:5000 mcstanleydocker27/youtubeinsights:latest

docker run -d -p 5000:5000 mcstanleydocker27/youtubeinsights:latest     # this continues to run the container after your local terminal has been shutdown

```
- push to dockerhub
```
docker login

docker push mcstanleydocker27/youtubeinsights:latest
```



# AWS CICD Deployment with GITHUB Actions

description:
```
# with specific access

1. EC2 access

2. ECR: Elastic Container Registry to save our docker image in aws

# Description About the deployment

1. Build docker image of the source code

2. Push the docker image to ECR

3. Launch EC2

4. Pull image from ECR in EC2

5. Launch docker image in EC2

# Policy:

1. AmazonEC2ContainerRegistryFullAccess

2. AmazonEC2FullAccess
```

## create for deployment NEW: 
    - IAM user (with policies listed above), 

    - ECR (to store docker images): save URI: 111624651583.dkr.ecr.af-south-1.amazonaws.com/yt_plugin_project

    - EC2 instance (Ubuntu with security groups 5000 port) and 
    - S3 bucket

## setup EC2
```
# optional

sudo apt-get update -y

sudo apt-get upgrade

# required

curl -fsSL https://get.docker.com -o get-docker.sh

sudo sh get-docker.sh

sudo usermod -aG docker ubuntu

newgrp docker
```

- configure as self hosted runner in github:

`settings>actions>runner>new self hosted runner> choose os> then run command one by one

- setup github secrets:
```
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION=
AWS_ECR_LOGIN_URI=
ECR_REPOSITORY_NAME=