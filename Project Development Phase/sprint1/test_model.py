import keras
from keras.models import load_model
import cv2
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model=load_model('model.h5')
from skimage.transform import resize
def detect(frame):
    img=resize(frame,(64,64,3))
    img=np.expand_dims(img,axis=0)
    if(np.max(img)>1):
        img = img/255.0
    prediction = model.predict(img)
    print(prediction)
    predict_x=model.predict(img)
    classes_x=np.argmax(predict_x,axis=1)
    print(classes_x)
frame=cv2.imread(r"C:\Users\Akshaya\PycharmProjects\Realtime_Communication_System_For_Specially_Abled\Dataset\asl_alphabet_test\asl_alphabet_test\D_test.jpg")
data=detect(frame)
