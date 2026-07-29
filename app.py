import cv2
import mediapipe as mp
import joblib
import numpy as np
import pandas as pd
import time

# ==========================
# Load Model
# ==========================
model = joblib.load("models/knn_model.pkl")

# Feature names
feature_names = pd.read_csv(
    "dataset/dataset.csv"
).drop("label", axis=1).columns

# ==========================
# Mudra Information
# ==========================
mudra_info = {
    "Pataka": {
        "meaning": "Flag",
        "uses": "Clouds, Forest, River"
    },
    "Mushti": {
        "meaning": "Fist",
        "uses": "Strength, Anger"
    },
    "Tripataka": {
        "meaning": "Three Parts of a Flag",
        "uses": "Crown, Tree, Arrow"
    },
    "Ardhachandra": {
        "meaning": "Half Moon",
        "uses": "Moon, Blessing, Plate"
    },
    "Hamsasya": {
        "meaning": "Swan Beak",
        "uses": "Pearl, Delicate Objects"
    }
}

# ==========================
# MediaPipe Setup
# ==========================
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# ==========================
# Webcam
# ==========================
cap = cv2.VideoCapture(0)


cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

prev_time = time.time()
fps = 0

last_console_prediction = ""

# ==========================
# Main Loop
# ==========================
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

        for lm in hand.landmark:
            landmarks.extend([
                lm.x,
                lm.y,
                lm.z
            ])

        input_data = pd.DataFrame(
            [landmarks],
            columns=feature_names
        )

        prediction = model.predict(input_data)[0]

        confidence = np.max(
            model.predict_proba(input_data)
        ) * 100

        # Confidence Threshold
        if confidence < 70:
            prediction = "Unknown Mudra"

        # Print only when prediction changes
        if prediction != last_console_prediction:
            if prediction != last_prediction:
                print(f"{prediction} ({confidence:.1f}%)")
                last_prediction = prediction
                last_console_prediction = prediction

    # ==========================
    # UI Panel
    # ==========================
    cv2.rectangle(frame, (10,10), (500,210), (45,45,45), -1)

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

    # ==========================
    # Confidence Bar
    # ==========================
    bar_x = 20
    bar_y = 120
    bar_width = 250
    bar_height = 20

    cv2.rectangle(
        frame,
        (bar_x, bar_y),
        (bar_x + bar_width, bar_y + bar_height),
        (255, 255, 255),
        2
    )

    filled_width = int((confidence / 100) * bar_width)

    if confidence >= 90:
        bar_color = (0, 255, 0)

    elif confidence >= 70:
        bar_color = (0, 255, 255)

    else:
        bar_color = (0, 0, 255)

    cv2.rectangle(
        frame,
        (bar_x, bar_y),
        (bar_x + filled_width, bar_y + bar_height),
        bar_color,
        -1
    )

    # ==========================
    # Mudra Meaning
    # ==========================
    if prediction in mudra_info:

        cv2.putText(
            frame,
            f"Meaning: {mudra_info[prediction]['meaning']}",
            (20, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Uses: {mudra_info[prediction]['uses']}",
            (20, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

    # ==========================
    # FPS
    # ==========================
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
    cv2.putText(
    frame,
    "Press Q to Exit",
    (950,700),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (255,255,255),
    2
)
    cv2.imshow(
        "AI Mudra Recognition System",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ==========================
# Cleanup
# ==========================
cap.release()
cv2.destroyAllWindows()