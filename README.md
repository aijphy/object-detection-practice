# object-detection-practice

This repository is used to practice object detection using pytorch, following chapter 7 of the book "Modern Computer Vision with PyTorch"

The goal is to train R-CNN model and put into production via AWS with an S3 bucket, Lambda, and REST API. The model should be 
accessible via a POST request with an image body, and return an image with the objects detected.

In order to save space, the dataset was not included in the repository.
The dataset used can be downloaded from:
https://www.kaggle.com/sixhky/open-images-bus-trucks
unzip the file in a separe directory (../open-images-bus-trucks/) to running the training module. Change the path in IMAGE_ROOT and DF_RAW if you choose a different directory name or location.


Alternatively, in order to avoid costly training, I use a readily available YOLO model https://github.com/fmacrae/YOLO-on-Lambda. The repo is already designed to run on Lambda. I have created a localserve.py file to allow the model to run on a local server.


In order to run the model on a local server:
copy or clone the localserve.py file
git clone https://github.com/fmacrae/YOLO-on-Lambda or just the unzip the darknet.zip file
pip install fastapi uvicorn aiofiles jinja2 pillow
wget https://pjreddie.com/media/files/yolov3.weights

uvicorn localserve:app

use postman (url example: http://127.0.0.1:8000/predict)
headers: 
    accept : application/json
    Content-Type : multipart/form-data
body:
    file : selected jpeg file

Alternatively, curl can be used with the same headers & body.

The jpg file you send will be saved, then used to generate a predictions.jpg file which displays the YOLO predictions.


