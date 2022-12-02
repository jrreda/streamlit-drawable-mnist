import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np


st.title("MNIST Digit Recognizer Customizable MLP")

### Model Hyperprameters
num_neurons = st.sidebar.slider("Number of neurons in hidden layer", 1, 128)
num_epochs = st.sidebar.slider("Number of epoches", 1, 64)
activation_fn = st.sidebar.selectbox("Activation function",
    ('relu', 'tanh', 'sigmoid', 'softmax', 'softplus',
    'softsign', 'selu', 'elu', 'exponential'))
optimizer_c = st.sidebar.selectbox("Optimizer",
    ('sgd', 'rmsprop', 'adam', 'adadelta', 'adagrad',
    'adamax', 'nadam', 'ftrl'))

st.write("The number of neurons:", str(num_neurons))
st.write("The number of epochs:", str(num_epochs))
st.write("The activation function:", str(activation_fn))
st.write("The Optimizer:", str(optimizer_c))


## load the data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# standerize the data
X_train = X_train / 255.0
X_test  = X_test / 255.0

## Create the model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(num_neurons, activation_fn),
    Dense(10, activation='softmax')
])
# Compile the model
model.compile(optimizer=optimizer_c, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

if st.button("Train the Model"):
    # train the model
    cp = ModelCheckpoint('model', save_best_only=True)
    history_cp=tf.keras.callbacks.CSVLogger('history.csv', separator=",", append=False)   #save to .csv
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=num_epochs, callbacks=[cp, history_cp])

    st.write("Model has been trained successfully")

if st.button("Evaluate the model"):    
    # read the saved callback
    history = pd.read_csv('history.csv')
    fig = plt.figure()
    
    plt.plot(history['epoch'], history['accuracy'])
    plt.plot(history['epoch'], history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Val'], loc='lower right')

    # Show the figure
    st.pyplot(fig)    


# DRAW AN INTEGER FROM 0 - 9
SIZE = 192

canvas_result = st_canvas(
    fill_color='#000000',
    stroke_width=10,
    stroke_color='#FFFFFF',
    background_color='#000000',
    height=SIZE,
    width=SIZE,
    drawing_mode='freedraw',
    key='canvas'
)

if canvas_result.image_data is not None:    
    img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
    rescaled = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    st.write('Model Input')
    st.image(rescaled)

if st.button("Predict"):
    pic_test = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pred = model.predict(pic_test.reshape(1, 28, 28, 1))
    st.write(f'result: {np.argmax(pred[0])}')
    st.bar_chart(pred[0])
        


# pip freeze > requirements.txt