import os
from glob import glob 
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import configs as cf
import numpy as np 


def get_data(meta_data=None, 
             batch_size=32,
             augmentation=True):
    if augmentation:
        data_gen = ImageDataGenerator(
        rescale=1/255.,
        brightness_range=[0.8,1.2],
        zoom_range=[1.0,1.2],
        horizontal_flip=True)
    else:
        data_gen = ImageDataGenerator(rescale=1/255.)

    train_generator = data_gen.flow_from_dataframe(
        meta_data,
        directory="./",
        x_col="image_path",
        y_col="label",
        class_mode="categorical",
        batch_size = batch_size,
        target_size=(cf.IMAGE_SIZE, cf.IMAGE_SIZE),
    )
    return train_generator


def get_meta_data(base_dir):
    # Tạo Metadata lưu giữ thông tin về image path và label 
    list_path = []
    list_labels = []
    for label in cf.CLASSES:
        label_path = os.path.join(base_dir, label, "*")
        image_files = glob(label_path)

        sign_label = [label] * len(image_files)

        list_path.extend(image_files)
        list_labels.extend(sign_label)

    meta_data = pd.DataFrame({
        "image_path": list_path,
        "label": list_labels
    })
    return meta_data

def preprocessing_image(image):
    img = np.array(img) / 255.
    img = img.reshape((1, cf.IMAGE_SIZE, cf.IMAGE_SIZE, 3))
    return img