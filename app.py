def greet(name):
    return "Hello " + name

title = "Deep Learning For Cervical  Cancer Lesion Segmentation In Mobile Colposcopy Images"
description = """
The model  was trained to segment cervical lesions and make informed prediction on images taken  using mobile Colposcopy!
"""
article = "Check out [the github repository](https://github.com/image-segmentation-for-cervical-cancer) that this website and model are based off of."

import gradio as gr
import cv2, math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import normalize
from tensorflow.keras.models import load_model

def IoU(y_true,y_pred):
  smooth= 1e-5
  y_true_f = float (K.flatten(y_true))
  y_pred_f = float (K.flatten(y_pred))
  intersection = K.sum(y_true_f*y_pred_f)
  result = (intersection + smooth)/(K.sum(y_true_f)+K.sum(y_pred_f)-intersection+ smooth)
  return result
def IoU_Loss(y_true,y_pred):
    smooth= 1e-5
    iou_loss = (1 - IoU(y_true,y_pred))
    return iou_loss
def get_config(self):
    config = super(IoU_Loss, self).get_config()
    config.update({'IoU_Loss': self.IoU_Loss})
    return config
    
model = tf.keras.models.load_model('my_model1 (2).h5',custom_objects={'IoU_Loss': IoU_Loss}, compile = False)
model.compile(optimizer='adam', 
              loss=IoU_Loss,
               metrics=[IoU,'binary_accuracy'])

def predict(input_img):
   
    #input_img = tf.image.resize(input_img, [256,256])
    input_img = input_img.reshape((512, 512, 3))
    test_img_input = normalize(input_img, axis=0)
    test_img_input= np.expand_dims(input_img, axis=0)
    prediction = model.predict(test_img_input)
    prediction = prediction[0,:,:,0]

    return prediction
    
examples = [
            ['image_test_data.jpg']
]

result = gr.Interface(predict,inputs=gr.Image(shape=(512, 512)), outputs=gr.Image(shape=(512, 512)), title=title, description=description, article=article, examples=examples)
result.launch()
