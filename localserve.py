from __future__ import print_function
import os, io
import subprocess
from PIL import Image
from fastapi import FastAPI, Request, File, UploadFile
import shutil
import uvicorn



app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hellow World"}

@app.post("/predict")
def predict(request:Request,file:UploadFile=File(...)):
    output = {"message": "running post"}
#    print("running...")
    with open("inputimg.jpg","wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        output["end"] = "done"
        imgfilepath = 'inputimg.jpg'
    
    strWeightFile = 'data/yolov3.weights'
    
    command = './darknet detect cfg/yolov3.cfg {} {}'.format(
        strWeightFile,
        imgfilepath
    )

    try:
        print('Start')
        o1 = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
        print('Finish')
        print(o1)
    except subprocess.CalledProcessError as e:
        print('Error')
        print(e.o1)
        output['error'] = 'error'

    return output

 # at last, the bottom of the file/module


