import cv2
import mediapipe as mp
import joblib
import numpy as np
import time
import pandas as pd

# Load trained model
model = joblib.load("models/knn_model.pkl")

# Get feature names from the training dataset
feature_names = pd.read_csv("dataset/dataset.csv").drop("label", axis=1).columns

prev_time = time.time()
fps = 0

mudra_info = {
    "Pataka": {
        "meaning": "Flag",
        "uses": "Clouds, Forest, River"
    },
    "Mushti": {
        "meaning": "Fist",
        "uses": "Strength, Anger"
    }
}

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    prediction = "Place your hand in view"
    confidence = 0

    if results.multi_hand_landmarks:

        hand = results.multi_hand_landmarks[0]

        mp_draw.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )

        landmarks = []

        # Wrist landmark
        for lm in hand.landmark:
            landmarks.extend([
        lm.x,
        lm.y,
        lm.z
         ])

        input_data = pd.DataFrame([landmarks], columns=feature_names)

        prediction = model.predict(input_data)[0]

        confidence = np.max(model.predict_proba(input_data)) * 100

        print(f"{prediction} ({confidence:.1f}%)")

    # Background panel
    cv2.rectangle(frame, (10, 10), (470, 180), (40, 40, 40), -1)

    cv2.putText(
        frame,
        "AI Mudra Recognition",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Mudra: {prediction}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Confidence: {confidence:.1f}%",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    if prediction in mudra_info:
        cv2.putText(
            frame,
            f"Meaning: {mudra_info[prediction]['meaning']}",
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Uses: {mudra_info[prediction]['uses']}",
            (20, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2
    )
    cv2.imshow("AI Mudra Recognition System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()