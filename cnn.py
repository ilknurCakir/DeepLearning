## importing libraries

from keras.models import Sequential
from keras.layers import Dense, Flatten, MaxPooling2D, Conv2D, Activation

# RGB colorchannels_last
# building the model
model = Sequential()
model.add(Conv2D(32, kernel_size = (3,3), input_shape = (64, 64, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(32, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())

model.add(Dense(128, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))

model.add(Dense(1, activation = 'sigmoid'))

# compiling the model
from keras import optimizers

optimizer = optimizers.Adam(lr = 0.01)
model.compile(optimizer = optimizer,
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])

# ImageDataGenerator for data augmentation

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory('dataset/training_set',
                                                        target_size=(64, 64),
                                                        batch_size=32,
                                                        class_mode='binary')

test_generator = test_datagen.flow_from_directory('dataset/test_set',
                                                    target_size=(64, 64),
                                                    batch_size=32,
                                                    class_mode='binary')

model.fit_generator(train_generator,
                    steps_per_epoch=8000,
                    epochs=10,
                    validation_data= test_generator,
                    validation_steps = 2000)

# testing a single picture



