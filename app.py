from flask import Flask, render_template,  request, redirect, url_for, send_from_directory
from keras.models import load_model
from flask import Response
import cv2
import sys
import numpy
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import numpy as np
from keras.applications.imagenet_utils import preprocess_input
from datetime import datetime
import csv

app = Flask(__name__, template_folder='templates')
mod = ''

def __del__(self):
        self.video.release()

#def get_frame():
#    camera_port=0
#    camera=cv2.VideoCapture(camera_port) 
#
#    while True:
#        _, im = camera.read()
#        imgencode=cv2.imencode('.jpg',im)[1]
#        stringData=imgencode.tostring()
#        yield (b'--frame\r\n'
#            b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')

def g_get_img():
    model = load_model('model/garbage.h5')
    camera_port=0
    camera=cv2.VideoCapture(camera_port) 

    while True:
        _, im = camera.read()
        imgencode = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        img = cv2.resize(imgencode, (300, 300)).astype("float32")
        #print('Done')
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        y_prob = model.predict(x) 
        y_classes = y_prob.argmax(axis=-1)
        #print('foooooooo')
        if y_classes == 0:
            now = datetime.now()
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
            with open('garbage.csv','a') as fd:
                writer = csv.writer(fd)
                writer.writerow(dt_string.replace(',', ''))
        
        imgencode=cv2.imencode('.jpg',im)[1]
        stringData=imgencode.tostring()
        yield (b'--frame\r\n'
            b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')



def f_get_img():
    model = load_model('model/fire.h5')
    camera_port=0
    camera=cv2.VideoCapture(camera_port) 

    while True:
        _, im = camera.read()
        imgencode = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        img = cv2.resize(imgencode, (300, 300)).astype("float32")
        #print('Done')
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        y_prob = model.predict(x) 
        y_classes = y_prob.argmax(axis=-1)
        #print('foooooooo')
        if y_classes == 0:
            now = datetime.now()
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
            with open('fire.csv','a') as fd:
                writer = csv.writer(fd)
                writer.writerow(dt_string.replace(',', ''))
        
        imgencode=cv2.imencode('.jpg',im)[1]
        stringData=imgencode.tostring()
        yield (b'--frame\r\n'
            b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')



@app.route('/', methods=['GET', "POST"])
def index():
    # Main page
    return render_template('index.html')

@app.route('/feed', methods=['GET', 'POST'])
def feed():
    if request.method == 'POST':
        model_name = request.form['model']
        mod = model_name + '_video_feed'

        return render_template('feed.html', mod=mod)

@app.route('/garbage_video_feed')
def garbage_video_feed():
    return Response(g_get_img(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/fire_video_feed')
def fire_video_feed():
    return Response(f_get_img(),mimetype='multipart/x-mixed-replace; boundary=frame')




if __name__ == '__main__':
    app.run()