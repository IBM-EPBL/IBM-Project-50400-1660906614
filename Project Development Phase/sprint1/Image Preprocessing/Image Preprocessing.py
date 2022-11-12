from keras_preprocessing.image import ImageDataGenerator
test_path = 'Dataset/test_set'
train_path = 'Dataset/training_set'
train=ImageDataGenerator(rescale=1./255,zoom_range=0.2,shear_range=0.2,horizontal_flip=True)
test=ImageDataGenerator(rescale=1./255)
train_batches = train.flow_from_directory(directory=train_path, target_size=(64,64), class_mode='categorical', batch_size=300,shuffle=True,color_mode="grayscale")
test_batches = test.flow_from_directory(directory=test_path, target_size=(64,64), class_mode='categorical', batch_size=300, shuffle=True,color_mode="grayscale")
