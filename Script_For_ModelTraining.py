from ultralytics import YOLO

# Load the YOLO model
model = YOLO('yolov8n.pt')  # Nano model, fast and lightweight

# Define the image path
image_path = '22.jpeg'

# Perform image detection
results = model(image_path, show=True)

# Save the results
for result in results:
    result.save('runs/detect/')  # Specify the folder to save results


# Alternatively, training the YOLOv8 model
# Initialize the YOLOv8 model
model = YOLO('yolov8n.pt')  # Nano model, fast and lightweight

# Start training the model
model.train(data='data.yaml', epochs=10, imgsz=640)


from ultralytics import YOLO

# Load the pre-trained model
model = YOLO('runs/detect/train/weights/best.pt')  # Or yolov8n.pt (pre-trained)

# Retrain the model
model.train(
    data='data.yaml',    # Dataset configuration file
    epochs=10,           # Number of training epochs
    imgsz=640,           # Image size in pixels
    weights='runs/detect/train/weights/best.pt'  # Pre-trained weights file
)

