import cv2
import mediapipe as mp
import numpy as np
import time

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)
canvas = None
prev_x, prev_y = 0, 0
draw_color = (255, 0, 0)
thickness = 5
last_save_time = 0

palette = {
    "Blue": ((20, 10), (70, 60), (255, 0, 0)),
    "Green": ((90, 10), (140, 60), (0, 255, 0)),
    "Red": ((160, 10), (210, 60), (0, 0, 255)),
    "Eraser": ((230, 10), (300, 60), (0, 0, 0))
}

def get_finger_tip(hand_landmarks, img_shape):
    h, w, _ = img_shape
    x = int(hand_landmarks.landmark[8].x * w)
    y = int(hand_landmarks.landmark[8].y * h)
    return x, y

def count_fingers(hand_landmarks):
    tips_ids = [8, 12, 16, 20]
    count = 0
    for tip in tips_ids:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            count += 1
    return count

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # Draw color palette
    for name, ((x1, y1), (x2, y2), color) in palette.items():
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        cv2.putText(frame, name, (x1 + 5, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            x, y = get_finger_tip(handLms, frame.shape)
            finger_count = count_fingers(handLms)

            # Color selection
            if y < 65:
                for name, ((x1, y1), (x2, y2), color) in palette.items():
                    if x1 < x < x2 and y1 < y < y2:
                        draw_color = color
                        thickness = 20 if name == "Eraser" else thickness

            elif finger_count == 1:  # Draw
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = x, y
                cv2.line(canvas, (prev_x, prev_y), (x, y), draw_color, thickness)
                prev_x, prev_y = x, y
            else:
                prev_x, prev_y = 0, 0

            # Increase/Decrease brush size
            if finger_count == 2 and thickness < 50:
                thickness += 1
                time.sleep(0.1)  # prevent too fast change
            if finger_count == 3 and thickness > 1:
                thickness -= 1
                time.sleep(0.1)

            # Clear canvas
            if finger_count == 4:
                canvas = np.zeros_like(frame)

    # Merge drawings
    combined = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
    cv2.putText(combined, f'Thickness: {thickness}', (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    cv2.imshow("Virtual Painter", combined)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        filename = f"drawing_{int(time.time())}.png"
        cv2.imwrite(filename, canvas)
        print(f"Saved as {filename}")

cap.release()
cv2.destroyAllWindows()
