import numpy as np
import tensorflow as tf
import cv2

from keras.models import Sequential, load_model

from flask import Flask, render_template, request, Response, redirect, url_for ,flash, session

from gtts import gTTS

from skimage.transform import resize

global graph
global writer
global pred

graph = tf.compat.v1.get_default_graph()
writer = None
pred = ''

model = load_model('aslpng1.h5')

vals = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

app = Flask(__name__)

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/index')
def home():
    return render_template('index.html')

camera = cv2.VideoCapture(0)

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def detect(frame):
    img = resize(frame,(64,64,1))
    img = np.expand_dims(img,axis=0)
    if(np.max(img)>1):
        img = img/255.0
    prediction = np.argmax(model.predict(img))
    print('\n\n'+vals[prediction]+'\n\n')
    return vals[prediction]

@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/capture')
def capture():
     success, frame = camera.read()
     cv2.imwrite('img.jpg', frame)
     text = detect(frame)
     pred = 'pred' + text
     return render_template("trial.html", info=text, comu=pred)

@app.route('/trial')
def trial():
    return render_template('trial.html',methods=["GET","POST"])

if (__name__ == '__main__'):
    app.run(debug = True)