import shutil
import os
import random
from imutils import paths
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.core import Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import SeparableConv2D
from keras.layers.core import Activation
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from keras.utils import np_utils


class myCallback(tf.keras.callbacks.Callback): # Callback to stop training when it reaches the desired accuracy
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>0.95):
            print("\nReached 95% accuracy so cancelling training!")
            self.model.stop_training = True

def data_build(source,destination,folder):	# This function is used to build the dataset 
    file_name = []
    for files in os.listdir(source):
        for file_0 in os.listdir(source+'/'+files+'/'+folder):
            file_name.append(source+'/'+files+'/'+folder+'/'+file_0)
    
    #print(len(file_name))
    random.shuffle(file_name)
    length_0 = int(len(file_name)*0.9)
    file_list = file_name[:length_0]
    length_1 = int(len(file_list)*0.9)
    file_list_training = file_list[0:length_1]
    file_list_validation = file_list[length_1:]
    file_list_testing = file_name[length_0:]
    
    for file_names in file_list_training:
        c = file_names.split('/')
        shutil.copy2(file_names, destination+'/training/'+folder+'/'+c[-1])
    for file_names in file_list_validation:
        c = file_names.split('/')
        shutil.copy2(file_names, destination+'/validation/'+folder+'/'+c[-1])
    for file_names in file_list_testing:
        c = file_names.split('/')
        shutil.copy2(file_names, destination+'/testing/'+folder+'/'+c[-1])


def model_1():		# This is the first simple model
    model = Sequential()
    model.add(Convolution2D(64,(3,3),input_shape=(50,50,3),activation = 'relu', padding="same"))
    model.add(MaxPooling2D((2,2)))
    model.add(Convolution2D(64,(3,3),activation = 'relu', padding="same"))
    model.add(MaxPooling2D((2,2)))
    model.add(Convolution2D(64,(3,3),activation = 'relu', padding="same"))
    model.add(MaxPooling2D((2,2)))
    model.add(Convolution2D(64,(3,3),activation = 'relu', padding="same"))
    model.add(MaxPooling2D((2,2)))
    model.add(Convolution2D(64,(3,3),activation = 'relu', padding="same"))
    model.add(MaxPooling2D((2,2)))
    model.add(Flatten())
    model.add(Dense(512,activation='relu'))
    model.add(Dense(2,activation='softmax'))
    return model

def model_2():		# This is the second model 
    model = Sequential()
    model.add(SeparableConv2D(32, (3,3), padding="same",input_shape=(50,50,3)))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(SeparableConv2D(64, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=-1))
    model.add(SeparableConv2D(64, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(SeparableConv2D(128, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=-1))
    model.add(SeparableConv2D(128, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=-1))
    model.add(SeparableConv2D(128, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(2))
    model.add(Activation("softmax"))
    return model

def model_4():		# This is the third model
    model = Sequential()
    model.add(Convolution2D(16,(3,3),input_shape=(50,50,3),activation = 'relu', padding="same"))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(32,(3,3),activation = 'relu', padding="same"))
    #model.add(MaxPooling2D((2,2)))

    model.add(Convolution2D(64,(3,3),activation = 'relu', padding="same"))
    model.add(BatchNormalization(axis=-1))
    #model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(128,(3,3),activation = 'relu', padding="same"))
    #model.add(MaxPooling2D((2,2)))

    model.add(Convolution2D(256,(3,3),activation = 'relu', padding="same"))
    #model.add(MaxPooling2D((2,2)))

    model.add(Convolution2D(512,(3,3),activation = 'relu', padding="same"))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(512,(3,3),activation = 'relu', padding="same"))
    model.add(BatchNormalization(axis=-1))
    #model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(512,(3,3),activation = 'relu', padding="same"))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512,activation='relu'))
    model.add(Dense(2,activation='softmax'))
    return model
    
    
def train_model(train_path, test_path, val_path, model, callbacks):	# This function trains the model
    train_datagen = ImageDataGenerator(
            rescale=1/255.0,
	          rotation_range=20,
	          zoom_range=0.05,
	          width_shift_range=0.1,
	          height_shift_range=0.1,
	          shear_range=0.05,
	          horizontal_flip=True,
	          vertical_flip=True,
	          fill_mode="nearest"
            )
    val_datagen = ImageDataGenerator(rescale=1./255)
    training = train_datagen.flow_from_directory(
            train_path,
            class_mode="categorical",
            target_size=(50,50),
            batch_size = 32,
            color_mode="rgb",
            shuffle=True,
            )
    validation = val_datagen.flow_from_directory(
	          val_path,
	          class_mode="categorical",
	          target_size=(50,50),
	          color_mode="rgb",
	          shuffle=False,
	          batch_size=32)
    testing = val_datagen.flow_from_directory(
            test_path,
            class_mode="categorical",
            target_size=(50,50),
            batch_size = 32,
            color_mode="rgb",
            shuffle=False,
            )
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics = ["accuracy"])
    m = model.fit_generator(
            training,
            epochs = 40,
            validation_data = validation,
            callbacks=[callbacks])
    testing.reset()
    pred_indices=model.predict_generator(testing)

    pred_indices=np.argmax(pred_indices,axis=1)

    print(classification_report(testing.classes, pred_indices, target_names=testing.class_indices.keys()))

    cm=confusion_matrix(testing.classes,pred_indices)
    total=sum(sum(cm))
    accuracy=(cm[0,0]+cm[1,1])/total
    specificity=cm[1,1]/(cm[1,0]+cm[1,1])
    sensitivity=cm[0,0]/(cm[0,0]+cm[0,1])
    print(cm)
    print(f'Accuracy: {accuracy}')
    print(f'Specificity: {specificity}')
    print(f'Sensitivity: {sensitivity}')

    N = 40
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0,N), m.history["loss"], label="train_loss")
    plt.plot(np.arange(0,N), m.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0,N), m.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0,N), m.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy on the IDC Dataset")
    plt.xlabel("Epoch No.")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig('plot.png')
    
    
    
if __name__ == '__main__':
    try:                    # please comment the whole try block if the directories already exists
        os.mkdir('/datasets')
        os.mkdir(os.getcwd()+'/datasets/dataset_1')
        os.mkdir(os.getcwd()+'/datasets/dataset_1/training')
        os.mkdir(os.getcwd()+'/datasets/dataset_1/testing')
        os.mkdir(os.getcwd()+'/datasets/dataset_1/validation')
        os.mkdir(os.getcwd()+'/datasets/dataset_1/validation/0')
        os.mkdir(os.getcwd()+'/datasets/dataset_1/validation/1')
        os.mkdir(os.getcwd()+'/datasets/dataset_1/training/0')
        os.mkdir(os.getcwd()+'/datasets/dataset_1/training/1')
        os.mkdir(os.getcwd()+'/datasets/dataset_1/testing/1')
        os.mkdir(os.getcwd()+'/datasets/dataset_1/testing/0')
    except:
        pass
    data_build(os.getcwd()+'/datasets/original/7415_10564_bundle_archive',os.getcwd()+'/datasets/dataset_1','0')
    data_build(os.getcwd()+'/datasets/original/7415_10564_bundle_archive',os.getcwd()+'/datasets/dataset_1','1')
    
    
    model = model_1()
    callbacks = myCallback()
    train_model(os.getcwd()+'/datasets/dataset_1/training/',os.getcwd()+'/datasets/dataset_1/testing/',os.getcwd()+'/datasets/dataset_1/validation',model,callbacks)
    
    """  Please uncomment this if you want to use model 4
    
    model_no_4 = model_4()
    callbacks = myCallback()
    train_model(os.getcwd()+'/datasets/dataset_1/training/',os.getcwd()+'/datasets/dataset_1/testing/',os.getcwd()+'/datasets/dataset_1/validation',model_no_4,callbacks)
    #model 4 ends here
    """
    
    """ Please uncomment this if you want to use model 2
    
    model2 = model_2()
    callbacks = myCallback()
    train_model(os.getcwd()+'/datasets/dataset_1/training/',os.getcwd()+'/datasets/dataset_1/testing/',os.getcwd()+'/datasets/dataset_1/validation',model2,callbacks)
    #model 2 ends here
    """
    
    """ Please Uncomment this if you want to use pretrained vgg model
    
    from keras import applications
    from keras.layers import Input
    from keras.layers.core import Dropout
    from keras.models import Model
    from keras.layers.convolutional import Conv2D

    input_tensor = Input(shape=(50, 50, 3))
    vgg_model = applications.VGG16(weights='imagenet',
                               include_top=False,
                               input_tensor=input_tensor)
    #print(vgg_model)
    layer_dict = dict([(layer.name, layer) for layer in vgg_model.layers])
    x = layer_dict['block2_pool'].output
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(2, activation='softmax')(x)
    custom_model = Model(inputs=vgg_model.input, outputs=x)
    for layer in custom_model.layers[:7]:
        layer.trainable = False
        #custom_model.compile(loss='categorical_crossentropy',
                             #optimizer='rmsprop',
                             #metrics=['accuracy'])
    train_model(os.getcwd()+'/datasets/dataset_1/training/',os.getcwd()+'/datasets/dataset_1/testing/',os.getcwd()+'/datasets/dataset_1/validation',custom_model,callbacks)
    #vgg model ends here
"""
