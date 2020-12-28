Web Application to detect faces and determine whether they are  wearing mask.

It's implemented on flask and deep learning framwork PyTorch

## How to run

### web server 
```
python app.py
```
web server runs on 5000 port

### mask detection on images

Get request to the server with parameter 'path' containing URl to a picture 

Response contains a json with a list of detected faces (whether they where mask or not, confidence and coordinates). 

If you need visualise the result add parameter show=True

e.g.

http://localhost:5000/api/?show=True&path=https://ww2.kqed.org/app/uploads/sites/10/2020/02/GettyImages-1198381294-800x512.jpg

### mask detection on video

Get request to the server with parameters 'mode=0' and 'path' containing URl to a video stream (0 for built-in camera) 

If you need visualise the result add parameter show=True

e.g.

http://localhost:5000/api/?mode=0&show=True&path=0

To stop video streaming

http://localhost:5000/stop/
