NO FUNCIONA LA PERRA MAMADA AAA

'''
import cv2
import mediapipe as mp
import os

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
	static_image_mode=False,
	max_num_hands=2,
	min_detection_confidence=0.5) as hands:


	while True:
		ret, frame = cap.read()
		if ret == False:
			break

		height, width, _ * frame.shape
		frame = cv2.flip(frame,1)

		cv2.imshow("Frame", frame)
		if cv2.waitKey(1) & 0xFF == 27:
			break

cap.release()
cv2.destroyAllWindows()
'''