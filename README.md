# object-detection-practice

This repository is used to practice object detection using pytorch, following chapter 7 of the book "Modern Computer Vision with PyTorch"

The goal is to train R-CNN model and put into production via AWS with an S3 bucket, Lambda, and REST API. The model should be 
accessible via a POST request with an image body, and return an image with the objects detected.

In order to save space, the dataset was not included in the repository.
The dataset used can be downloaded from:
https://www.kaggle.com/sixhky/open-images-bus-trucks
unzip the file in a separe directory (../open-images-bus-trucks/) to running the training module. Change the path in IMAGE_ROOT and DF_RAW if you choose a different directory name or location.

