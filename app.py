from flask import Flask, request, Response,render_template,send_file
import time
from flask_cors import CORS
from flask import Flask, render_template, request
import cv2
import requests
import json
import os
import numpy as np
import argparse
import pickle
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from os import listdir
from PIL import Image
from numpy import asarray
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN

from os import listdir
from os.path import isdir
from PIL import Image
from matplotlib import pyplot
from numpy import savez_compressed
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from numpy import load
from numpy import expand_dims
from numpy import asarray
from numpy import savez_compressed
from keras.models import load_model
import pickle

PATH_TO_TEST_IMAGES_DIR = './images'


app = Flask(__name__)
CORS(app)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))


def get_embedding(model, face_pixels):
    face_pixels = face_pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    samples = expand_dims(face_pixels, axis=0)
    yhat = model.predict(samples)
    return yhat[0]


def extract_face(filename, required_size=(160, 160)):
    image = Image.open(filename)
    image = image.convert('RGB')
    pixels = asarray(image)
    detector = MTCNN()
    results = detector.detect_faces(pixels)
    x1, y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = pixels[y1:y2, x1:x2]
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array

def load_faces(directory):
    faces = list()
    for filename in listdir(directory):
        path = directory + filename
        face = extract_face(path)
        faces.append(face)
    return faces

def load_dataset(directory):
    
    X, y = list(), list()
    for subdir in listdir(directory):
        print(subdir)
        path = directory + subdir + '/'
        if not isdir(path):
            continue
        faces = load_faces(path)
        labels = [subdir for _ in range(len(faces))]
        print('>loaded %d examples for class: %s' % (len(faces), subdir))
        X.extend(faces)
        y.extend(labels)
    return asarray(X), asarray(y)



def detect(path,file):
    model = load_model('facenet_keras.h5')
    in_encoder=pickle.loads(open("in_encoder.pickle","rb").read())
    out_encoder=pickle.loads(open("out_encoder.pickle","rb").read())
    model1=pickle.loads(open("recognizer.pickle","rb").read())
    face_orig=extract_face(path)
    faces=get_embedding(model,face_orig)
    x=in_encoder.transform([faces])
    pred=model1.predict_proba(x)
    classes=np.argmax(pred[0])
    probability=pred[0][classes]
    classes=out_encoder.inverse_transform([classes])[0]
    title = '%s (%.3f)' % (classes, probability*100)
    pyplot.title(title)
    pyplot.savefig("static/"+file)
    
"""@app.route('/')
def index():
    return render_template("getImage.html")
    #return Response(open('./static/getImage.html').read(), mimetype="text/html")
"""
# save the image as a picture
@app.route('/image', methods=['POST'])
#@crossdomain(origin='*',headers=['access-control-allow-origin','Content-Type'])
def image():

    i = request.files['image']  # get the image
    f = ('%s.jpeg' % time.strftime("%Y%m%d-%H%M%S"))
    i.save('%s/%s' % (PATH_TO_TEST_IMAGES_DIR, f))
    image=cv2.imread(PATH_TO_TEST_IMAGES_DIR+"/"+f)##reading the image from html form
    #image = imutils.resize(image, width=600)
    
    print("writing image")##checkpoint for debugging errors
    detect(image,"1"+f)##The actual Detection function
    return send_file("static/"+filename, mimetype='image/gif')
    #return render_template("results.html",image_name=f)
    #return Response("%s saved" % f)

if __name__ == '__main__':
    app.run()
