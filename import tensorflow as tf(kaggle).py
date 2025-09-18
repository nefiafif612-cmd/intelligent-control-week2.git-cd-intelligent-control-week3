import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Direktori dataset
train_dir = "D:\NEFI AFIF SUJATYANA\FILE KULIAH\SEMESTER 7\Prak. KONTROL CERDAS\M3\seg_train"
val_dir = "D:\NEFI AFIF SUJATYANA\FILE KULIAH\SEMESTER 7\Prak. KONTROL CERDAS\M3/seg_test"

# Augmentasi data
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 150), batch_size=32, class_mode='categorical')
val_generator = val_datagen.flow_from_directory(val_dir, target_size=(150, 150), batch_size=32, class_mode='categorical')