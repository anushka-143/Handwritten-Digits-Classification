import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import keras
import keras.datasets.mnist
import pandas as pd
from keras import layers

(x_train,y_train),(x_test,y_test) = keras.datasets.mnist.load_data()
#scale the values or normalize the values between 0 and 1
x_train = x_train/255
x_test = x_test/255
'''plt.matshow(x_train[2])
plt.show()'''

'''Flattening the training and test dataset : converting the data into a 1-dimensional array for inputting it to the next layer'''
#print(x_train.shape)
x_train_F = x_train.reshape(len(x_train),28*28)
x_test_F = x_test.reshape(len(x_test),28*28)

print(x_train_F[0])

#creating a simple neural network model
model = keras.Sequential([
    layers.Dense(100,input_shape=(784,),activation='relu'),
    layers.Dense(10,activation='sigmoid')
])
model.compile(optimizer='Adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train_F,y_train,epochs=5)
model.evaluate(x_test_F,y_test)

#creating variable for predicted values
y_pred = model.predict(x_test_F)

# check the first image in x_test[0]
plt.matshow(x_test[0])
plt.show()

#priting the predicted scores, there are 10 score which represents 0 to 9 respectively
print(y_pred[0])

#lets print the index of the maximum score
print(np.argmax(y_pred[0]))

#we need to convert y_predicted as they are whole values into integers so for that we will use list comprehension
y_pred_labels = [np.argmax(i) for i in y_pred]

cm = tf.math.confusion_matrix(labels=y_test,predictions=y_pred_labels)
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot = True,fmt = 'd')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()