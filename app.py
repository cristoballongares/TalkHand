
import cv2
import mediapipe as mp
import os

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

with mp_hands.Hands(
	static_image_mode=True,
	max_num_hands=2,
	min_detection_confidence=0.5) as hands:

	image = cv2.imread("image.png")
	width, height = (image.shape, image.shape)
	image = cv2.flip(image, 1)

	image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	results = hands.process(image_rgb)


	print("Handedness: ", results.multi_handedness)

	if results.multi_hand_landmarks is not None:

		for hand_landmarks in results.multi_hand_landmarks:
			#print(hand_landmarks)
			mp_drawing.draw_landmarks(
				image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
				mp_drawing.DrawingSpec(color=(255,255,0), thickness=4, circle_radius=5),
				mp_drawing.DrawingSpec(color=(255,0,255), thickness=4, circle_radius=5))


	image = cv2.flip(image, 1)
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.DestroyAllWindows()