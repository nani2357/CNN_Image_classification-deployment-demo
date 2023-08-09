
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import os
import random

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer




# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))

classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

base_dir = r'E:\DATA\dog_cat_CNN\train'
dog_dir = os.path.join(base_dir, 'dog')
cat_dir = os.path.join(base_dir, 'cat')


# Get all dog and cat image filenames
all_dog_images = os.listdir(dog_dir)
all_cat_images = os.listdir(cat_dir)

# Shuffle the lists
random.shuffle(all_dog_images)
random.shuffle(all_cat_images)


# Select the first 20% of the shuffled list
selected_dog_images = all_dog_images[:int(len(all_dog_images) * 0.2)]
selected_cat_images = all_cat_images[:int(len(all_cat_images) * 0.2)]


# Define new directories for the selected subset
subset_base_dir = r'E:\DATA\dog_cat_CNN\subset_train'
subset_dog_dir = os.path.join(subset_base_dir, 'dog')
subset_cat_dir = os.path.join(subset_base_dir, 'cat')

# Create the directories if they don't exist
os.makedirs(subset_dog_dir, exist_ok=True)
os.makedirs(subset_cat_dir, exist_ok=True)


import shutil

# Copy selected dog images to the new directory
for img_name in selected_dog_images:
    src_path = os.path.join(dog_dir, img_name)
    dst_path = os.path.join(subset_dog_dir, img_name)
    shutil.copy(src_path, dst_path)

# Copy selected cat images to the new directory
for img_name in selected_cat_images:
    src_path = os.path.join(cat_dir, img_name)
    dst_path = os.path.join(subset_cat_dir, img_name)
    shutil.copy(src_path, dst_path)
    
    
datagen = ImageDataGenerator(validation_split=0.2,
                                               rescale = 1./255,
                                                shear_range = 0.2,
                                                zoom_range = 0.2,
                                                horizontal_flip = True)
  


training_set = datagen.flow_from_directory(
    subset_base_dir,
    target_size=(64, 64),  
    batch_size=32,
    class_mode='binary',
    subset='training',  # Set as training data
    shuffle=True,
    seed=42
)


test_set = datagen.flow_from_directory(
    subset_base_dir,
    target_size=(64, 64),  
    batch_size=32,
    class_mode='binary',
    subset='validation',  # Set as training data
    shuffle=True,
    seed=42
)


model = classifier.fit(training_set,
                         steps_per_epoch = 125,
                         epochs = 10,
                         validation_data = test_set,    
                         validation_steps = 31)


print(training_set.class_indices)
print(test_set.class_indices)

classifier.save("model.h5")
print("Saved model to disk")


from keras.models import load_model

loaded_model = load_model("model.h5")



import numpy as np
from keras.preprocessing import image
test_image = image.load_img(r'E:\DATA\dog_cat_CNN\test1\test1\11.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = loaded_model.predict(test_image)

if result[0][0] == 1:
    prediction = 'dog'
    print(prediction)
else:
    prediction = 'cat'
    print(prediction)

