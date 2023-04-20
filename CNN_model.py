#Importing necessary libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l2




#Setting up the variables
num_features = 64
num_labels = 7
batch_size = 128
epochs = 80
width, height = 48, 48


#Loading the dataset and labels
x = np.load('FER_ROBOTICS_CW2/fdataX.npy')
y = np.load('FER_ROBOTICS_CW2/flabels.npy')


#Preprocessing the data
x -= np.mean(x, axis=0)
x /= np.std(x, axis=0)


#splitting into training, validation and testing data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=41)

#saving the test samples to be used later
np.save('FER_ROBOTICS_CW2/modXtest', X_test)
np.save('FER_ROBOTICS_CW2/modytest', y_test)

#Designing the CNN

#This code creates a sequential model for the CNN and assigns it to the variable CNN_Fer_model.
CNN_Fer_model = Sequential()

#1. CNN_Fer_model.add(): This function adds layers to the sequential model.
#2. Conv2D(): This function creates a 2D convolutional layer. 
#       The parameters passed to this function specify the number of features, 
#       kernel size, activation function, input shape, padding, and regularization.
#3. BatchNormalization(): This function normalizes the activations of the previous layer.
#4. MaxPooling2D(): This function applies max pooling operation for spatial data.
#5. Dropout(): This function applies dropout regularization to the layer.

CNN_Fer_model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1), data_format='channels_last', kernel_regularizer=l2(0.01)))
CNN_Fer_model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', padding='same'))
CNN_Fer_model.add(BatchNormalization())
CNN_Fer_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
CNN_Fer_model.add(Dropout(0.5))

CNN_Fer_model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
CNN_Fer_model.add(BatchNormalization())
CNN_Fer_model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
CNN_Fer_model.add(BatchNormalization())
CNN_Fer_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
CNN_Fer_model.add(Dropout(0.5))

CNN_Fer_model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
CNN_Fer_model.add(BatchNormalization())
CNN_Fer_model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
CNN_Fer_model.add(BatchNormalization())
CNN_Fer_model.add(Dropout(0.5))

CNN_Fer_model.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
CNN_Fer_model.add(BatchNormalization())
CNN_Fer_model.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
CNN_Fer_model.add(BatchNormalization())
CNN_Fer_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
CNN_Fer_model.add(Dropout(0.5))

#Flatten(): This function flattens the input.
CNN_Fer_model.add(Flatten())

#Dense(): This function creates a dense layer with the specified number of neurons,
#       activation function, and dropout regularization.

CNN_Fer_model.add(Dense(2*2*2*num_features, activation='relu'))
CNN_Fer_model.add(Dropout(0.4))
CNN_Fer_model.add(Dense(2*2*num_features, activation='relu'))
CNN_Fer_model.add(Dropout(0.4))
CNN_Fer_model.add(Dense(2*num_features, activation='relu'))
CNN_Fer_model.add(Dropout(0.5))

CNN_Fer_model.add(Dense(num_labels, activation='softmax'))

#CNN_Fer_model.summary(): This function prints the summary of the model, 
#                         including the number of layers, number of parameters, and shape of each layer.
CNN_Fer_model.summary()

#Compliling the model with adam optimixer and categorical crossentropy loss
CNN_Fer_model.compile(loss=categorical_crossentropy,
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
              metrics=['accuracy'])


#training the model
CNN_Fer_model.fit(np.array(X_train), np.array(y_train),
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(np.array(X_valid), np.array(y_valid)),
          shuffle=True)


#saving the  model to be used later


#=======================================
# Below model produced 78% accuracy
#         
#fer_json = CNN_Fer_model.to_json()
#with open("FER_ROBOTICS_CW2/emotion_model.json", "w") as json_file:
#    json_file.write(fer_json)
#CNN_Fer_model.save_weights("FER_ROBOTICS_CW2/emotion_model.h5")
#print("Saved model to disk")
#=======================================

#=======================================
# Below model produced 71% accuracy
#         
fer_json = CNN_Fer_model.to_json()
with open("FER_ROBOTICS_CW2/FER_CNN_Model.json", "w") as json_file:
    json_file.write(fer_json)
CNN_Fer_model.save_weights("FER_ROBOTICS_CW2/FER_CNN_Model.h5")
print("Saved model to disk")
#=======================================