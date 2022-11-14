import keras
from keras.models import load_model
import cv2
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
val=['A','B','C','D','E','F','G','H','I']

model=load_model('model.h5')
from skimage.transform import resize
def detect(frame):
    img=resize(frame,(64,64,1))
    img=np.expand_dims(img,axis=0)
    if(np.max(img)>1):
        img = img/255.0
    predict_x=model.predict(img)
    print(predict_x)
    predict=np.argmax(predict_x,axis=1)
    x=predict[0]
    print(val[x])
frame=cv2.imread(r"C:\Users\Akshaya\PycharmProjects\Realtime_Communication_System_For_Specially_Abled\Dataset\test_set\B\1.png")
data=detect(frame)
