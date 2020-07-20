## About the project
### Major Components
1. Dockerfile
2. requirement.txt
3. Model File
4. Jenkins Jobs for automation.
 
#### 1. Dockerfile
First we'll create a Dockerfile to create a Docker Image  to setup environment with all dependencies installs.

Dockerfile looks like this.

```Dockerfile
FROM python:3.6

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
```

#### 2. requirements.txt
It has all liabraies listed for all environments (Keras, Sci-Kit Learn, Tensorflow)

Dependencies for Keras can be found  [here]( Requirements\Keras\requirements.txt "Keras")

Dependencies for TensorFlow can be found  [here]( Requirements\TensorFlow\requirements.txt "TensorFlow")

Dependencies for Sci-Kit Learn can be found  [here]( Requirements\SkLearn\requirements.txt "Sci-Kit Learn")

#### 2. Model File
Model file has these major components.

##### Liabraries
```python
from keras.datasets.mnist import load_data
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, ReLU, Dropout
from keras.layers import BatchNormalization, MaxPooling2D
from keras.utils import to_categorical
from keras.callbacks import Callback, ModelCheckpoint
import numpy as np
from os import getenv, system
```
##### Helper Functions
function to load dataset (by default is load keras mnist digit dataset)
```python
def load_dataset():
    (X_train, Y_train), (X_test, Y_test) = load_data(path='mnist.npz')
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
    Y_train = to_categorical(Y_train)
    Y_test = to_categorical(Y_test)
    return X_train, X_test, Y_train, Y_test
```
function to preprocess dataset.
```python
def Prep_dataset(X_train, X_test):
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train = X_train/255.0
    X_test = X_test/255.0
    return X_train, X_test

```

function to initialize Model.
```python
def model_init():
    model = Sequential()
    model.add(Conv2D(filters=512,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding="same",
                     activation='relu',
                     input_shape=(28, 28, 1)
                     ))
    return model
```
function to add one conv block. One block contains
Conv2D -> BatchNormalization -> MaxPooling2D -> Dropout -> ReLU
```python
def add_block(model):
    model.add(Conv2D(filters=256,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding="valid",
                     activation='relu'
                     ))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(ReLU())
    return model
```
function to finalise the model  flatten it then 2 dense layer with dropout and last sofmax function.
```python
def finalise(model):
    model.add(Flatten())
    model.add(Dense(units=64, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    return model
```

##### Environmental variables
```python
no_blocks    = getenv('NO_BLOCKS')
batch_size   = getenv('BATCH_SIZE')
epochs       = getenv('EPOCHS')
filepath     = getenv('FILEPATH')
```
##### Model creation with helper function.
```python
model = model_init()

for i in range(1, no_blocks+1):
    model = add_block(model)

model = finalise(model)

model.compile(optimizer='Adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
print(model.summary())

```
##### Callbacks and Fitting the model
```python
X_train, X_test, Y_train, Y_test = load_dataset()
X_train, X_test = Prep_dataset(X_train, X_test)


checkpoint = ModelCheckpoint(
    filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

earlyStopping = EarlyStopping(
    monitor='val_accuracy',
    min_delta=0.01,
    patience=0,
    verbose=0,
    mode='auto',
    baseline=None,
    restore_best_weights=True)

callbacks_list = [checkpoint, earlyStopping]
history = model.fit(x=X_train,
          y=Y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(X_test, Y_test),
          shuffle=True,
          callbacks=callbacks_list,
          use_multiprocessing=True)
val_acc = history.history['val_accuracy'][-1]
system('export VAL_ACC = val_acc')
```

> Link to Full `model.py` find 
> 
> [here](model.py "Model")

#### 4. Jenkins Jobs for automation.