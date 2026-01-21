# import time
# import cv2
# import cvzone
# from ultralytics import YOLO

# # Confidence threshold
# confidence = 0.6

# # Start webcam
# cap = cv2.VideoCapture(0)
# cap.set(3, 640)
# cap.set(4, 480)

# # Load the trained model
# model = YOLO("best (2).pt")

# # Class names
# classNames = ["bottle", "crushed_bottle", "none"]

# # Threshold height (you can tweak this based on your visuals)
# CRUSHED_HEIGHT_THRESHOLD = 350  # Pixels

# prev_frame_time = 0

# while True:
#     new_frame_time = time.time()
#     success, img = cap.read()
#     if not success:
#         print("Failed to grab frame")
#         break

#     # Run inference
#     results = model(img, stream=True, verbose=False)

#     for r in results:
#         for box in r.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             w, h = x2 - x1, y2 - y1
#             conf = float(box.conf[0])
#             if conf < confidence:
#                 continue

#             # Simulate class using bounding box height
#             if h < CRUSHED_HEIGHT_THRESHOLD:
#                 cls = 1  # crushed_bottle
#             else:
#                 cls = 0  # bottle

#             label = classNames[cls]

#             # Color coding
#             if label == "bottle":
#                 color = (0, 255, 0)
#             elif label == "crushed_bottle":
#                 color = (255, 165, 0)
#             else:
#                 color = (0, 0, 255)

#             # Draw bounding box and label
#             cvzone.cornerRect(img, (x1, y1, w, h), colorC=color, colorR=color)
#             cvzone.putTextRect(
#                 img,
#                 f'{label.upper()} {int(conf * 100)}%',
#                 (max(0, x1), max(35, y1)),
#                 scale=1.5,
#                 thickness=2,
#                 colorR=color,
#                 colorB=color
#             )

#     # Display frame
#     cv2.imshow("Bottle Detector", img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

import time
import cv2
import cvzone
from ultralytics import YOLO

# Confidence threshold
confidence = 0.6

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Load the trained model
model = YOLO("Final.pt")

# Class names — ensure these match your training!
classNames = ["bottle", "crushed_bottle", "none"]

while True:
    success, img = cap.read()
    if not success:
        print("Failed to grab frame")
        break

    # Run inference
    results = model(img, stream=True, verbose=False)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            conf = float(box.conf[0])

            if conf < confidence:
                continue

            cls = int(box.cls[0])
            if cls < len(classNames):
                label = classNames[cls]
            else:
                label = "unknown"

            # Color coding
            if label == "bottle":
                color = (0, 255, 0)
            elif label == "crushed_bottle":
                color = (255, 165, 0)
            else:
                color = (0, 0, 255)

            # Draw bounding box and label
            cvzone.cornerRect(img, (x1, y1, w, h), colorC=color, colorR=color)
            cvzone.putTextRect(
                img,
                f'{label.upper()} {int(conf * 100)}%',
                (max(0, x1), max(35, y1)),
                scale=1.5,
                thickness=2,
                colorR=color,
                colorB=color
            )

    # Display frame
    cv2.imshow("Bottle Detector", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
