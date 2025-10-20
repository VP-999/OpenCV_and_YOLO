import cv2 as cv
import numpy as np

# Load Haar cascades
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_smile.xml')

# Initialize webcam
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

prev_frame = None

# Start video loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detect eyes
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.3, minNeighbors=5)
        for (ex, ey, ew, eh) in eyes:
            cv.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        # Detect smile
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)
        for (sx, sy, sw, sh) in smiles:
            cv.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)

        # Eye blink detection
        if len(eyes) == 0:
            cv.putText(frame, "Eyes Closed!", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Distance measurement
        distance = 500 / w
        cv.putText(frame, f"Distance: {distance:.2f} cm", (x, y - 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Motion detection
    if prev_frame is not None:
        diff_frame = cv.absdiff(prev_frame, gray)
        _, thresh_frame = cv.threshold(diff_frame, 30, 255, cv.THRESH_BINARY)
        motion_detected = cv.countNonZero(thresh_frame) > 500
        if motion_detected:
            cv.putText(frame, "Motion Detected!", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    prev_frame = gray

    # Show frame
    cv.imshow('Face and Eye Detection - Master_sec', frame)

    # Save frame on 's' key press
    if cv.waitKey(1) & 0xFF == ord('s'):
        cv.imwrite("output.jpg", frame)
        print("Frame saved as output.jpg")

    # Exit on 'q' key press
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv.destroyAllWindows()
