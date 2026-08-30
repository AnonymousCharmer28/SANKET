import os
import csv
import cv2
import mediapipe as mp
import numpy as np

DATASET_DIR = r"E:\sanket\dataset"
CSV_PATH = "isl_features.csv"

def extract_landmarks():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5
    )

    header = ['label']
    for hand_type in ['left', 'right']:
        for i in range(21):
            header.extend([f'{hand_type}_x_{i}', f'{hand_type}_y_{i}', f'{hand_type}_z_{i}'])

    with open(CSV_PATH, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        
        image_count = 0
        print("Starting extraction. This will take a while...")

        # Walk through all subdirectories (Kids, Teenagers, etc.)
        for root, dirs, files in os.walk(DATASET_DIR):
            folder_name = os.path.basename(root).upper()
            
            # Map E1 and E2 to the letter E
            if folder_name in ['E1', 'E2']:
                label = 'E'
            else:
                label = folder_name
            
            # Only process if the folder represents a single English letter
            if len(label) == 1 and label.isalpha():
                for image_name in files:
                    if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        continue
                        
                    image_path = os.path.join(root, image_name)
                    img = cv2.imread(image_path)
                    
                    if img is None:
                        continue

                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    results = hands.process(img_rgb)
                    
                    features = np.zeros(126)

                    if results.multi_hand_landmarks and results.multi_handedness:
                        for idx, hand_handedness in enumerate(results.multi_handedness):
                            hand_label = hand_handedness.classification[0].label
                            offset = 0 if hand_label == 'Left' else 63
                            
                            for i, lm in enumerate(results.multi_hand_landmarks[idx].landmark):
                                features[offset + (i * 3)] = lm.x
                                features[offset + (i * 3) + 1] = lm.y
                                features[offset + (i * 3) + 2] = lm.z

                    writer.writerow([label] + features.tolist())
                    
                    image_count += 1
                    if image_count % 1000 == 0:
                        print(f"Processed {image_count} images...")

    hands.close()
    print(f"Extraction complete. Total images processed: {image_count}. Data saved to {CSV_PATH}")

if __name__ == "__main__":
    extract_landmarks()