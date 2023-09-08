import cv2
import numpy as np 
from keras.models import load_model
import src.configs as cf 


model = load_model('./model/cnn_asl_model.h5')

def regconize():
	cam = cv2.VideoCapture(0)
	text = ""
	word = ""
	count_same_frame = 0
	while True:
		frame = cam.read()[1]
		# Target area where the hand gestures should be.
		cv2.rectangle(frame, (0, 0), (cf.CROP_SIZE, cf.CROP_SIZE), (0, 255, 0), 3)
		# Preprocessing the frame before input to the model.
		cropped_image = frame[0:cf.CROP_SIZE, 0:cf.CROP_SIZE]
		resized_frame = cv2.resize(cropped_image, (cf.IMAGE_SIZE, cf.IMAGE_SIZE))
		reshaped_frame = (np.array(resized_frame)).reshape((1, cf.IMAGE_SIZE, cf.IMAGE_SIZE, 3))
		frame_for_model = reshaped_frame/255.0
		old_text = text
		prediction = np.array(model.predict(frame_for_model))
		prediction_probability = prediction[0, prediction.argmax()]
		text = cf.CLASSES[prediction.argmax()]      # Selecting the max confidence index.
		if text == 'space':
			text = '_'
		if text != 'nothing':	
			if old_text == text:
				count_same_frame += 1
			else:
				count_same_frame = 0

			if count_same_frame > 10:
				word = word + text
				count_same_frame = 0
		blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
		cv2.putText(blackboard, " ", (180, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 0,0))
		cv2.putText(blackboard, f"Predict: {text}", (30, 100), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 0))
		cv2.putText(blackboard, "Probability: {:.2f}%".format(prediction_probability * 100), (30, 170), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 0))
		cv2.putText(blackboard, word, (30, 300), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 255, 255))
		res = np.hstack((frame, blackboard))
		cv2.imshow("Recognizing gesture", res)
		k = cv2.waitKey(1) & 0xFF
		if k == ord('q'):
			break
		if k == ord('r'):
			word = ""
		if k == ord("z"):
			word = word[:-1]
regconize()