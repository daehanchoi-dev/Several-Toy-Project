import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras import layers
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

# Load Data

from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

"""
print(type(x_train))
print(type(y_train))
print(type(x_test))
print(type(y_test))
print('x_train shape:',x_train.shape)
print('y_train shape:',y_train.shape)
print('x_test shape:',x_test.shape)
print('y_test shape:',y_test.shape)
print(x_train[0])
print('\ntab\n')
print(y_train[0])
"""

classification = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 레이블을 원핫 인코딩으로 숫자로 치환해주기
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)
"""
print(y_train_one_hot)
print('\ntab\n')
print(y_test_one_hot)
"""

# 0과 1사이의 값으로 정규화하기
x_train = x_train / 255
x_test = x_test / 255

# 모델 생성

model = Sequential()
# 첫번째 레이어
model.add(Conv2D(32, (5,5), activation='relu', input_shape=(32,32,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
# 두번째 레이어
model.add(Conv2D(32, (5,5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(500, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(250, activation='relu'))

# 10개의 각각 다른 분류를 해야하기 때문에
model.add(Dense(10, activation='softmax'))

# 모델 컴파일

model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics= ['accuracy'])

# 모델 학습

hist = model.fit(x_train, y_train_one_hot,
                 batch_size= 256,
                 epochs= 10,
                 validation_split = 0.2)

# 모델 평가

model.evaluate(x_test, y_test_one_hot)[1]

# Accuracy 시각화 하기

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.savefig('Accuracy.png')
plt.show()

# Loss 시각화 하기

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.savefig('Loss.png')
plt.show()