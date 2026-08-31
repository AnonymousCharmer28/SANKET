import cv2
import mediapipe as mp
import joblib
import numpy as np
import warnings
from collections import deque
import statistics

warnings.filterwarnings("ignore")

def main():
    print("Booting SANKET AI Engine...")
    try:
        clf = joblib.load('models/rf_model.pkl')
        le = joblib.load('models/label_encoder.pkl')
    except FileNotFoundError:
        print("Critical Error: Model files not found.")
        return

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

    cap = cv2.VideoCapture(0)
    print("Camera active. Press 'q' to shut down.")

    # Initialize a buffer to stabilize the output
    prediction_buffer = deque(maxlen=10)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        features = np.zeros(126)

        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                if i > 1:
                    break
                
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                offset = i * 63
                for j, lm in enumerate(hand_landmarks.landmark):
                    features[offset + (j * 3)] = lm.x
                    features[offset + (j * 3) + 1] = lm.y
                    features[offset + (j * 3) + 2] = lm.z

            # Predict and add to buffer
            prediction = clf.predict([features])
            predicted_letter = le.inverse_transform(prediction)[0]
            prediction_buffer.append(predicted_letter)

            # Get the most common prediction in the last 10 frames
            try:
                smoothed_letter = statistics.mode(prediction_buffer)
            except statistics.StatisticsError:
                smoothed_letter = predicted_letter # Fallback if tie

            # Draw UI
            cv2.rectangle(frame, (20, 20), (150, 100), (15, 15, 15), -1)
            cv2.putText(frame, smoothed_letter, (55, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 127), 3)

        cv2.imshow('SANKET: Live ISL Translation MVP', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()