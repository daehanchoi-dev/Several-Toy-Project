from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras import applications
import numpy as np

# dimensions of images

img_width, img_height = 150, 150

VGG16_weights = 'vgg16_weights.h5'
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
train_samples = 2000
validation_samples = 800
epochs= 50
batch_size = 16


def BottleNeck_feature():
    datagen = ImageDataGenerator(rescale=1./255)

    # Build VGG Network
    model = applications.VGG16(include_top=False, weights='imagenet')

    train_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size = (img_width, img_height),
        batch_size = batch_size,
        class_mode='None',
        shuffle=False)
    feature_train = model.predict_generator(
        train_generator, train_samples // batch_size)
    np.save(open('feature_train.npy', 'wb'),
            feature_train)

    validation_generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size= (img_width, img_height),
        batch_size= batch_size,
        class_mode='None',
        shuffle=False)
    feature_validation = model.predict_generator(
        validation_generator, validation_samples // batch_size)
    np.save(open('feature_validation.npy', 'wb'),
            feature_validation)


def train_model():
    train_data = np.load(open('feature_train.npy', 'rb'))
    train_labels = np.array(
        [0] * (train_samples / 2) + [1]  * (train_samples / 2))

    validation_data = np.load(open('feature_validation.npy', 'rb'))
    validation_labels = np.array(
        [0] * (validation_samples / 2) + [1] * (validation_samples /2))

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss= 'binary_crossentropy',
                  metrics = ['accuracy'])
    model.fit(train_data, train_labels,
              epochs= epochs,
              batch_size=batch_size,
              validation_data = (validation_data, validation_labels))

    model.save_weights(VGG16_weights)


BottleNeck_feature()
train_model()
