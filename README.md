# object-detection-practice

This repository is used to practice object detection using pytorch, following chapter 7 of the book "Modern Computer Vision with PyTorch" as well as moving a model to production. Initially, a pytorch RCNN model was created (object_detection.py) but due to training cost, a YOLO model is being used in place to practice moving to production.

The goal is to train an object detection model and put it into production via AWS with an S3 bucket, Lambda, and REST API. The model should be 
accessible via a POST request with an image body, and return an image with the objects detected. Currently, a YOLO model is being served locally with localserve.py. The model was used on Lambda with an S3 trigger, but resulted in a timeout from the API.

I use a readily available YOLO model https://github.com/fmacrae/YOLO-on-Lambda. The repo is already designed to run on Lambda. I have created a localserve.py file to allow the model to run on a local server.

In order to run the model as a server:

run setup.sh

uvicorn localserve:app

use postman or curl to access the api: 
(url example: http://127.0.0.1:8000/predict)

headers: 

    accept : application/json
    
    Content-Type : multipart/form-data
    
body:

    file : selected jpeg file

The png file you send will be saved, then used to generate a predictions.jpg file which displays the YOLO predictions.

A Dockerfile and requirements.txt file were also included to make a docker image that contains the uvicorn server.

The docker image can be built and ran after running setup.sh with:

docker build -t 'name':latest .

docker run -p 8000:8000 'name':latest

Notes:
The dataset for the pytorch model was not included in the repository in order to save space.
The dataset can be downloaded from:
https://www.kaggle.com/sixhky/open-images-bus-trucks
unzip the file in a separate directory (../open-images-bus-trucks/) to running the training module. Change the path in IMAGE_ROOT and DF_RAW if you choose a different directory name or location.


