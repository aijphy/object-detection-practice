
# setup directory for uvicorn
git clone https://github.com/fmacrae/YOLO-on-Lambda
rm YOLO-on-Lambda/LICENSE*
rm YOLO-on-Lambda/README.md
rm YOLO-on-Lambda/serverless.yaml
rm YOLO-on-Lambda/service.py
mv YOLO-on-Lambda/* .
rm -rf YOLO-on-Lambda
cd data/labels
unzip labels.zip
cd ../../
chmod +x darknet
pip install fastapi uvicorn aiofiles jinja2 pillow python-multipart
wget https://pjreddie.com/media/files/yolov3.weights
mv yolov3.weights data/

# to run the uvicorn server:
# uvicorn localserve:app

# to setup and run docker container:
# docker build -t yolotest:latest .
# docker run -p 8000:8000 yolotest
