from PIL import Image
import numpy as np
import os
from glob import glob
import sys
from tempfile import TemporaryFile
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Activation
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import model_from_json
from sklearn.preprocessing import StandardScaler
px = 9

def loaddata():
    wayG = "C:/Users/Adrian/Desktop/inzynierka/GreenBox"
    wayR = "C:/Users/Adrian/Desktop/inzynierka/RedBox"
    os.chdir(wayG)
    file = glob("*.tif")
    X=np.zeros((len(file),px,px,3))
    y=np.zeros(len(file))

    for i in range(len(file)):
        try:
            img = Image.open(file[i])
            X[i]=np.array(img)
            y[i]=np.array(file[i][0])
            
                    
        except IOError: 
            pass
    return(X,y)

def CNN():
    
    X,y = loaddata()
    print(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    
    nb_of_classes = 3 
    y_train = keras.utils.to_categorical(y_train, nb_of_classes)
    y_test = keras.utils.to_categorical(y_test,  nb_of_classes)
    print(y_train)
    model = Sequential()
    model.add(Conv2D(32, (3, 3), data_format='channels_last',input_shape=(px, px, 3),padding='SAME'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), data_format='channels_last',input_shape=(px, px, 3),padding='SAME'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())  
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_of_classes))
    model.add(Activation('softmax'))
    
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    
    history = model.fit(X_train, y_train, epochs=100,  validation_split=0.20)
    
    scores = model.evaluate(X_test, y_test, batch_size=2)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

    way = "C:/Users/Adrian/Desktop/inzynierka/CNNjson"
    os.chdir(way)
    
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model.h5")
    print("Saved model to disk")
    
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'vali'], loc='upper right')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'vali'], loc='upper right')
    plt.show()

def Predict():
    
    X,y = loaddata()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    
    nb_of_classes = 3 
    y_train = keras.utils.to_categorical(y_train, nb_of_classes)
    y_test = keras.utils.to_categorical(y_test,  nb_of_classes)
    
    way = "C:/Users/Adrian/Desktop/inzynierka/FISH"
    wayjson = "C:/Users/Adrian/Desktop/inzynierka/CNNjson"
    
    wymiar = 1376*1032 #1376 1032
    img = np.empty(20580,dtype=object)
    Dot = np.empty(wymiar,dtype=object)
    
    
    os.chdir(wayjson)
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    loaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    score = loaded_model.evaluate(X_test, y_test, verbose=0)
    print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))     
    
    os.chdir(way)
    
    file = glob("FISH (21).tif")
    Dot = np.zeros(wymiar,dtype=object)
    for i in range(wymiar):
        Dot[i]=(0,0,0)
    
    for i in range(len(file)):
        
        try:
            img[i] = Image.open(file[i])
            w,h = img[i].size
                      
            for o in range(h-px-3):
                for j in range(w-px-3):
                    Box = (j, o, j+px, o+px)
                    X = np.array(img[i].crop(Box))
                    X = X.reshape(1,px,px,3)
                    point=loaded_model.predict(X)
                    if point[0][1] > 0.97:                                                     
                        Dot[(o+int(px/2))*w+j+int(px/2)] = (0,255,0)
                    elif point[0][2] > 0.97:                                                     
                        Dot[(o+int(px/2))*w+j+int(px/2)] = (255,0,0)
                          
            img[i].putdata(Dot)
            img[i].save("1.tif")
            
            os.chdir(way)       
        except IOError: 
                 pass
