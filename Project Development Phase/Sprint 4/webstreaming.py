import numpy as np
import tensorflow as tf
import cv2, os, sqlite3

from keras.models import Sequential, load_model

from flask import Flask, render_template, request, Response, redirect, url_for ,flash, session

from gtts import gTTS

global graph
global writer

from skimage.transform import resize

from cloudant.client import Cloudant

graph = tf.compat.v1.get_default_graph()
writer = None

app = Flask(__name__)
app.secret_key="123"

con=sqlite3.connect("database.db")
# con.execute("create table if not exists customer(pid integer primary key,name text, email text, number integer, password text)")
# con.close()

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/index')
def home():
    return render_template('index.html')

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/trial')
def trial():
    return render_template('trial.html')

camera = cv2.VideoCapture(0)

model = load_model('sld_model.h5')

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
            

vals = ['C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C',
        'C', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A',
        'C', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'C', 'C', 'C']

def detect(frame):
    img = resize(frame,(256,256,1))
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
     pred = ""
     pred += text
     return render_template("prediction.html", info=text, comu=pred)

# @app.route('/voice', methods=['GET','POST'])
# def voice():
    
#     return text

@app.route('/post_register', methods=["GET","POST"])
def post_register():
    if request.method=='POST':
        try:
            name=request.form['name']
            email=request.form['email']
            number=request.form['number']
            password=request.form['password']
            con=sqlite3.connect("database.db")
            cur=con.cursor()
            cur.execute("insert into customer(name, email, number, password)values(?,?,?,?)",(name,email,number,password))
            con.commit()
        #     flash("User created  Successfully","success")
        # except:
        #     flash("Error in Insert Operation","danger")
        finally:
            return redirect(url_for('login'))
            con.close()
    return render_template('register.html')

@app.route('/post_login', methods=["GET","POST"])
def post_login():
    if request.method=="POST":
        email=request.form['email']
        password=request.form['password']
        con=sqlite3.connect("database.db")
        con.row_factory=sqlite3.Row
        cur=con.cursor()
        cur.execute("select * from customer where email=? and password=?",(email, password))
        data=cur.fetchone()
        if data:
            session["email"]=data["email"]
            session["password"]=data["password"]
            return redirect("prediction")
    return redirect(url_for("login"))

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/history')
def hostory():
    return render_template('history.html')

@app.route('/logout')
def logout():
    session.clear()
    return render_template("logout.html")

if (__name__ == '__main__'):
    app.run(debug = True)