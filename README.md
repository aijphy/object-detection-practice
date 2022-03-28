# object-detection-practice

This repository is used to practice object detection using the book "Modern Computer Vision with PyTorch", as well as moving a model to production. Initially, a pytorch RCNN model was created (object_detection.py) but due to training cost, a YOLO darknet model is being used in place to practice moving to production.

The initial goal was to train an object detection model and put it into production via AWS with an S3 bucket, Lambda, and REST API. The model should be 
accessible via a POST request with an image body, and return an image with the objects detected. The YOLO model (https://github.com/fmacrae/YOLO-on-Lambda) was uploaded to Lambda. I attempted first to avoid S3 by using REST API which took an image body request and called the Lambda function. The model took longer than the API gateway timeout of 30 seconds, which cannot be modified. To get around the timeout, I created an S3 trigger when an image was uploaded (via an API post request) that called the Lambda function. When this occured, the Lambda function would be triggered twice (I'm still unsure of the reason. It is possible the image gets duplicated?).  In order to keep the focus on learning new tools instead of handling bugs, I changed the direction from AWS to local kubernetes.

Currently, the YOLO model is being served locally with uvicorn (localserve.py) and can be made into a Docker container and served via minikube following the steps below:

In order to run the model as a server:

    ./setup.sh

to serve without container:
    uvicorn localserve:app

use postman or curl to access the api: 
(url example: http://127.0.0.1:8000/predict)

headers: 

    accept : application/json
    Content-Type : multipart/form-data
    
body:

    file : selected jpeg file

The png file you send will be saved, then used to generate a predictions.jpg file which displays the YOLO predictions.

To make docker image:

The docker image can be built and ran after running setup.sh with:

    docker build -t 'name':latest .
    docker run -p 8000:8000 'name':latest


To run with kubernetes via minikube:
    
    minikube start
    eval $(minikube docker-env)
    docker build -t localserve .
    kubectl create -f kubernetes/pod.yaml

to test pod:
    
    kubectl port-forward client-pod 8000:80

can test in browser with http://localhost:8000/

create service:
    
    kubectl create -f kubernetes/nodeportservice.yaml
    kubectl port-forward service/client-node-port 8000:80
    can test in browser with http://localhost:8000/


Because it is using Docker, it does not expose the minikube ip to localhost, so port-forwarding is the easiest way to get around it locally. The kubernetes cluster can be deployed later via cloud.


Notes:
The dataset for the pytorch model was not included in the repository in order to save space.
The dataset can be downloaded from:
https://www.kaggle.com/sixhky/open-images-bus-trucks.
Unzip the file in a separate directory (../open-images-bus-trucks/) prior to training with object_detection.py. Change the path in IMAGE_ROOT and DF_RAW if you choose a different directory location.


