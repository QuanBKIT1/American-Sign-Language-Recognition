import cv2
import numpy as np
import configs as cf
from pipeline import preprocessing_image
def predict_image(model, image_path):
    img = cv2.imread(image_path)
    img = preprocessing_image(img)
    prediction = np.array(model.predict(img))
    predicted = cf.CLASSES[prediction.argmax()]
    print("Predict label: ", predicted)
