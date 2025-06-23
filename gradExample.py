import numpy as np
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import losses

m_test=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14])
n_test= m_test**2

def model_gen(Input_shape):
    X_input = Input(shape=Input_shape)
    X = Dense(units=128, activation='relu')(X_input)
    X = Dense(units=64, activation='relu')(X)
    X = Dense(units=32, activation='relu')(X)
    X = Dense(units=1)(X)
    model = Model(inputs=X_input, outputs=X)
    return model

model = model_gen(Input_shape=(1,))
opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.001)
model.compile(loss=losses.mean_squared_error, optimizer=opt)
model.fit(m_test,n_test, epochs=1000)


x = np.array([1,2,3,4,5,6])
x = tf.convert_to_tensor(x, np.float32)
with tf.GradientTape() as tape:
    tape.watch(x)
    y = model(x)
grad_y = tape.gradient(y, x)  
print(y)
print(grad_y)
