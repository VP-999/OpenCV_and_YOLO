from ultralytics import YOLO
import cv2
import os
from datetime import datetime

# Load the YOLOv8 pre-trained model
model = YOLO('yolov8n.pt')  
# Load camera feed or video file
video_path = 0  # Use 0 for live feed, or specify the path to a video file
cap = cv2.VideoCapture(video_path)

# Define class labels
classes = ["person", "knife", "gun", "fighting", "keyboard", "book", "mouse"]

# Create folders for saving screenshots and video clips
screenshot_folder = "screenshots_for_ML"
video_folder = "video_clips_for_ML"
os.makedirs(screenshot_folder, exist_ok=True)
os.makedirs(video_folder, exist_ok=True)

# Initialize video writer for saving clips
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = None
recording = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Process the frame using YOLOv8
    results = model.predict(frame, stream=True)

    # Loop through all detections
    harmful_activity_detected = False
    for box in results:
        if len(box.boxes) == 0:  # If no detection is found, skip to the next frame
            continue

        # Process detected boxes
        for detection in box.boxes:
            x1, y1, x2, y2 = map(int, detection.xyxy[0].tolist())  # Bounding box coordinates
            conf = detection.conf.item()  # Confidence score
            cls = int(detection.cls.item())  # Class ID

            # Get class label and confidence score
            if cls < len(classes):  # If the class ID is valid
                label = f"{classes[cls]} {conf:.2f}"

                # Draw the bounding box and label
                color = (0, 0, 255) if cls in [1, 2, 3] else (0, 0, 255)  # Red for harmful, green otherwise
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Mark harmful activity if detected
                if cls in [1, 2, 3]:  # Harmful activities: knife, gun, fighting
                    harmful_activity_detected = True

    # Annotate and save the image if harmful activity detected
    if harmful_activity_detected:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        annotated_path = os.path.join(screenshot_folder, f"annotated_{timestamp}.jpg")
        cv2.imwrite(annotated_path, frame)
        print(f"Annotated image saved as {annotated_path}")

        # Start recording video if not already recording
        if not recording:
            video_path = os.path.join(video_folder, f"clip_{timestamp}.avi")
            video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
            recording = True

        # Write the current frame to the video file
        if video_writer is not None:
            video_writer.write(frame)

    # Stop recording if no harmful activity is detected
    elif recording:
        print("Stopping video recording.")
        recording = False
        if video_writer is not None:
            video_writer.release()
            video_writer = None

    # Display the output
    cv2.imshow("Harmful Activity Detection", frame)

    # Save frame on 's' key press
    key = cv2.waitKey(1)
    if key & 0xFF == ord('s'):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_path = os.path.join(screenshot_folder, f"screenshot_{timestamp}.jpg")
        cv2.imwrite(screenshot_path, frame)
        print(f"Screenshot saved as {screenshot_path}")

    # Press 'q' to exit the video
    if key & 0xFF == ord('q'):
        print("Exiting program.")
        break

# Release resources
cap.release()
if video_writer is not None:
    video_writer.release()
cv2.destroyAllWindows()
