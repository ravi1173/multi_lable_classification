from __future__ import division, print_function
# coding=utf-8
import os
import sys

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import model_from_json
from keras.initializers import glorot_uniform


# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer


#tell our app where our saved model is
# sys.path.append(os.path.abspath("./model"))
# from load import *
#initalize our flask app
app = Flask(__name__)
#global vars for easy reusability
# global model, graph
#initialize these variables



# Load your trained model
# model = load_model('model.h5')
# model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
# from keras.applications.resnet50 import ResNet50
# model = ResNet50(weights='imagenet')
# model.save('')
# print('Model loaded. Check http://127.0.0.1:5000/')

def init(): 
    #Reading the model from JSON file
    with open('model.json', 'r') as json_file:
        loaded_model_json = json_file.read()
    #load the model architecture 
    loaded_model = model_from_json(loaded_model_json)

    loaded_model.load_weights("model.h5")

    #compile and evaluate loaded model
    loaded_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

    return loaded_model




def model_predict(img_path, model):
    classes = ['single_seater_sofa', 'double_seater_sofa', 'triple_seater_sofa', 'single_bed','double_bed']
    img = load_img(img_path, target_size=(224, 224, 3))
    plt.imshow(img)
    img = img_to_array(img)
    img = img/255.0
    img = img.reshape(1, 224, 224, 3)
    y_prob = model.predict(img)
    top_pred = np.argsort(y_prob[0])[-1]
    return (classes[top_pred], y_prob[0][top_pred])

def get_color_temp(r,g,b):
  n=( 0.23881*r+ 0.25499*g - 0.58291*b)/(0.11109*r - 0.85406*g+ 0.52289*b)
  return ((449*n**3)+(3525*n**2)+(6823.3*n)+(5520.33))

def get_pixel_value(test_img):
  color_image = Image.open(test_img)
  color_image_rgb = color_image.convert("RGB")
  rgb_pixel_value = color_image_rgb.getpixel((112,112))
  r,g,b = rgb_pixel_value
  temp = get_color_temp(r,g,b)
  return temp



@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.getcwd()
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        
        model = init()
        # Make prediction
        seats, prob = model_predict(file_path, model)
        temp = get_pixel_value(file_path)
        image_path = '..\\uploads\\' + f.filename

    return render_template('after.html', pred=seats, prob=prob, img_url=image_path, temp=temp)

if __name__ == '__main__':
    app.run(debug=True)