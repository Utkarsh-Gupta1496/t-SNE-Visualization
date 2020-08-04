"""
Created on Thu Dec  6 18:52:34 2019

@author: utkarsh
"""

import numpy as np 
import keras  
from keras.datasets import mnist 
from keras.models import Model 
from keras.layers import Dense, Input
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten 
from keras import backend as k 
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

(x_train, y_train), (x_test, y_test) = mnist.load_data()

c=np.random.randint(0,60000,500)
x_train= x_train[c]
y_train1= y_train[c]

y_train = keras.utils.to_categorical(y_train) 
y_test = keras.utils.to_categorical(y_test)
y_train= y_train[c]

img_rows, img_cols=28, 28
  
if k.image_data_format() == 'channels_first': 
   x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols) 
 #  x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols) 
   inpx = (1, img_rows, img_cols) 
  
else: 
   x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1) 
#   x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1) 
   inpx = (img_rows, img_cols, 1) 
  
x_train = x_train.astype('float32') 
#x_test = x_test.astype('float32') 
x_train /= 255
#x_test /= 255

inpx = Input(shape=inpx)  
layer1 = Conv2D(32, kernel_size=(3, 3), activation='relu',kernel_initializer='random_uniform',bias_initializer='zeros')(inpx) 
layer2 = Flatten()(layer1)
layer3 = Dense(128, activation='relu',kernel_initializer='random_uniform',bias_initializer='zeros')(layer2) 
layer4 = Dense(128, activation='relu',kernel_initializer='random_uniform',bias_initializer='zeros')(layer3) 
layer5 = Dense(10, activation='softmax',kernel_initializer='random_uniform',bias_initializer='zeros')(layer4)

model = Model([inpx], layer5) 
model.compile(optimizer=keras.optimizers.Nadam(), 
              loss=keras.losses.categorical_crossentropy, 
              metrics=['accuracy']) 
model.summary()

layer_output = []
for i in range(3,5):
  get_3rd_layer_output = k.function([model.layers[0].input],
                                  [model.layers[i].output])
  output = ((get_3rd_layer_output([x_train])[0]))
  layer_output.append(output)

X_embedded = TSNE(n_components=2).fit_transform(layer_output[0])

X_embedded.shape
vis_x = X_embedded[:, 0]
vis_y = X_embedded[:, 1]
plt.rcParams['figure.figsize'] = [10, 10]
plt.scatter(vis_x, vis_y, c=y_train1, cmap=plt.cm.get_cmap("jet", 10),marker='H')
plt.colorbar(ticks=range(10))
plt.clim(-0.5, 9.5)
plt.title('Before Training(Dense Layer 1 )')
plt.show()

X_embedded = TSNE(n_components=2).fit_transform(layer_output[1])

X_embedded.shape
vis_x = X_embedded[:, 0]
vis_y = X_embedded[:, 1]
plt.rcParams['figure.figsize'] = [10, 10]
plt.scatter(vis_x, vis_y, c=y_train1, cmap=plt.cm.get_cmap("jet", 10),marker='H')
plt.colorbar(ticks=range(10))
plt.clim(-0.5, 9.5)
plt.title('Before Training(Dense Layer 2)')
plt.show()



model.fit(x_train, y_train, epochs=30, batch_size=500)

layer_output = []
for i in range(3,5):
  get_3rd_layer_output = k.function([model.layers[0].input],
                                  [model.layers[i].output])
  output = ((get_3rd_layer_output([x_train])[0]))
  layer_output.append(output)

X_embedded = TSNE(n_components=2).fit_transform(layer_output[0])

X_embedded.shape
vis_x = X_embedded[:, 0]
vis_y = X_embedded[:, 1]
plt.rcParams['figure.figsize'] = [10, 10]
plt.scatter(vis_x, vis_y, c=y_train1, cmap=plt.cm.get_cmap("jet", 10),marker='H')
plt.colorbar(ticks=range(10))
plt.clim(-0.5, 9.5)
plt.title('After 30 iteration of  Training(Dense Layer 1 )')
plt.show()

X_embedded = TSNE(n_components=2).fit_transform(layer_output[1])

X_embedded.shape
vis_x = X_embedded[:, 0]
vis_y = X_embedded[:, 1]
plt.rcParams['figure.figsize'] = [10, 10]
plt.scatter(vis_x, vis_y, c=y_train1, cmap=plt.cm.get_cmap("jet", 10),marker='H')
plt.colorbar(ticks=range(10))
plt.clim(-0.5, 9.5)
plt.title('After 30 iteration of Training(Dense Layer 2)')
plt.show()

model.fit(x_train, y_train, epochs=30, batch_size=500)

layer_output = []
for i in range(3,5):
  get_3rd_layer_output = k.function([model.layers[0].input],
                                  [model.layers[i].output])
  output = ((get_3rd_layer_output([x_train])[0]))
  layer_output.append(output)

X_embedded = TSNE(n_components=2).fit_transform(layer_output[0])

X_embedded.shape
vis_x = X_embedded[:, 0]
vis_y = X_embedded[:, 1]
plt.rcParams['figure.figsize'] = [10, 10]
plt.scatter(vis_x, vis_y, c=y_train1, cmap=plt.cm.get_cmap("jet", 10),marker='H')
plt.colorbar(ticks=range(10))
plt.clim(-0.5, 9.5)
plt.title('After Complete Training(60 ITERATION)(Dense Layer 2)')
plt.show()

X_embedded = TSNE(n_components=2).fit_transform(layer_output[1])

X_embedded.shape
vis_x = X_embedded[:, 0]
vis_y = X_embedded[:, 1]
plt.rcParams['figure.figsize'] = [10, 10]
plt.scatter(vis_x, vis_y, c=y_train1, cmap=plt.cm.get_cmap("jet", 10),marker='H')
plt.colorbar(ticks=range(10))
plt.clim(-0.5, 9.5)
plt.title('After Complete Training(60 ITERATION)(Dense Layer 2)')
plt.show()
