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
# 인덱스 순서... 인덱스 0 번 부터 비행기, 자동차, 새, 고양이 ..... 9번은 트럭
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

# 예제를 통한 모델 테스트
import cv2

Image = cv2.imread('cat.jpg')
cv2.imshow('Cute_cat', Image)
cv2.waitKey()

# 이미지를 입력하기 위해 크기 조정

from skimage.transform import resize

resized_image = resize(Image, (32,32,3))
cv2.imshow('Cute_Cat', resized_image)


# 모델을 통한 예측

prediction = model.predict(np.array([resized_image]))
print(prediction)

# 이미지의 유사도가 큰 순서대로 인덱스 정렬하기

index = [0,1,2,3,4,5,6,7,8,9]
x = prediction

for i in range(10):
    for j in range(10):
        if x[0][index[i]] > x[0][index[j]]:
            idx = index[i]
            index[i] = index[j]
            index[j] = idx

print(index)

# 가장 유사한 5개의 인덱스

for i in range(5):
    print(classification[index[i]], ':', round(prediction[0][index[i]] * 100, 2), '%')
