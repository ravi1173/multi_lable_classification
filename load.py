import numpy as np
#import keras.models
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import model_from_json
#from scipy.misc import imread, imresize,imshow
#import tensorflow as tf
from keras.initializers import glorot_uniform
    

#tf.compat.v1.disable_eager_execution()
#tf.compat.v1.get_default_graph()

def init(): 
    #Reading the model from JSON file
    with open('model.json', 'r') as json_file:
        loaded_model_json = json_file.read()
    #load the model architecture 
    loaded_model = model_from_json(loaded_model_json)

    loaded_model.load_weights("model.h5")

    #compile and evaluate loaded model
    loaded_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    
    # classes = ['single_seater_sofa', 'double_seater_sofa', 'triple_seater_sofa', 'single_bed','double_bed']
    # img = load_img(file_path, target_size=(224, 224, 3))
    # plt.imshow(img)
    # img = img_to_array(img)
    # img = img/255.0
    # img = img.reshape(1, 224, 224, 3)
    # y_prob = loaded_model.predict(img)
    # top_pred = np.argsort(y_prob[0])[-1]
    # return (classes[top_pred], y_prob[0][top_pred])
    
    #loss,accuracy = model.evaluate(X_test,y_test)
    #print('loss:', loss)
    #print('accuracy:', accuracy)
    #graph = tf.get_default_graph()

    return loaded_model