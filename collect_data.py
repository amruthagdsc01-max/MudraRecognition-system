import cv2
import mediapipe as mp
import csv
import os
import sys

# Usage:
# python collect_data.py Pataka

if len(sys.argv) != 2:
    print("Usage: python collect_data.py <MudraName>")
    exit()

label = sys.argv[1]

os.makedirs("dataset", exist_ok=True)
csv_file = "dataset/dataset.csv"

# Create CSV header if it doesn't exist
if not os.path.exists(csv_file):
    header = ["label"]
    for i in range(21):
        header.extend([f"x{i}", f"y{i}", f"z{i}"])

    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

print(f"Collecting data for: {label}")
print("Press C to capture")
print("Press Q to quit")

sample_count = 0

while True:
    success, frame = cap.read()

    if not success:
        print("Failed to read from webcam.")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

    cv2.putText(
        frame,
        f"{label} | Samples: {sample_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("Dataset Collector", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("c") and results.multi_hand_landmarks:
        row = [label]

        for lm in hand_landmarks.landmark:
            row.extend([lm.x, lm.y, lm.z])

        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

        sample_count += 1
        print(f"Saved sample {sample_count}")

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()