import pickle

import cv2
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(1)



mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'A', 1: 'E', 2: 'I', 3: 'O', 4: 'U'}
while True:




    x_left = []
    y_left = []
    x_right = []
    y_right = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    ret, frame = cap.read()
    height, width, _ = frame.shape
    text = "Pulse ESC para salir"
    org = (10, height - 10)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1
    thickness = 2
    color = (255, 0, 0)
    bg_color = (0, 0, 255)
    rect_height = 40

        # Obtener el tamaño del texto
    text_size, _ = cv2.getTextSize(text, font, scale, thickness)

        # Definir las coordenadas del rectángulo
    rect_x1 = org[0]
    rect_y1 = org[1] - text_size[1]
    rect_x2 = org[0] + text_size[0]
    rect_y2 = org[1] - text_size[1] + rect_height

    # Dibujar el rectángulo y el texto
    cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), bg_color, -1)
    cv2.putText(frame, text, org, font, scale, color, thickness, cv2.LINE_AA)

    # Verifica cuántas manos se detectaron
    if results.multi_hand_landmarks:
        if len(results.multi_hand_landmarks) == 1:  # Una mano detectada
            hand_landmarks_1 = results.multi_hand_landmarks[0]
            # Resto del código que procesa las características de una mano
            for i in range(len(hand_landmarks_1.landmark)):
                x = hand_landmarks_1.landmark[i].x
                y = hand_landmarks_1.landmark[i].y

                x_left.append(x)
                y_left.append(y)

            x1 = int(min(x_left) * W) - 10
            y1 = int(min(y_left) * H) - 10

            x2 = int(max(x_left) * W) - 10
            y2 = int(max(y_left) * H) - 10

            data_aux = []
            for i in range(len(hand_landmarks_1.landmark)):
                x = hand_landmarks_1.landmark[i].x
                y = hand_landmarks_1.landmark[i].y

                x_left.append(x)
                y_left.append(y)

                # Agregar solo una vez cada landmark
                data_aux.append(x - min(x_left))
                data_aux.append(y - min(y_left))

            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

        
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break



cap.release()
cv2.destroyAllWindows()