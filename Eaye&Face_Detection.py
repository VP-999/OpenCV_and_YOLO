
import cv2 as cv

# Load the Haar cascades for face and eye detection
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')

# Open webcam video capture
cap = cv.VideoCapture(0)  #here 0 is the default camera

if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

# Start video loopc
while True:
    # Capture frame-by-frame from the webcam
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert frame to grayscale for Haar cascade detection
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Loop through detected faces
    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Region of Interest (ROI) for eye detection within the face
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detect eyes in the face region
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.3, minNeighbors=5)

        # Loop through detected eyes
        for (ex, ey, ew, eh) in eyes:
            # Draw rectangle around the eyes
            cv.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    # Display the resulting frame with face and eye detection
    cv.imshow('Eye Detection - Webcam - Nazim', frame)

    # Exit loop on 'q' key press
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv.destroyAllWindows()
