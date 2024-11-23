import cv2  # OpenCV for image and video processing
import numpy as np  # NumPy for array operations
import mediapipe as mp  # Mediapipe for pose estimation
import tensorflow as tf  # TensorFlow for enabling GPU memory growth
from ultralytics import YOLO  # YOLOv8 for object detection

# Initialize GPU for TensorFlow
# List all available GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')

# Enable dynamic memory growth for each GPU
# This prevents TensorFlow from allocating all GPU memory at once
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Initialize Mediapipe pose estimation
mp_pose = mp.solutions.pose  # Access Mediapipe's pose solution
pose = mp_pose.Pose()  # Create a Mediapipe Pose object for landmark detection

# Load the YOLOv8 model for object detection
# 'yolov8n.pt' is a pretrained YOLOv8 nano model
# The `.to('cuda')` ensures the model runs on the GPU for faster inference
model = YOLO('yolov8n.pt').to('cuda')

# Define colors for pose connections (based on the rainbow)
CONNECTION_COLORS = [
    (0, 0, 255),       # Red
    (0, 165, 255),     # Orange
    (0, 255, 255),     # Yellow
    (0, 255, 0),       # Green
    (255, 0, 0),       # Blue
    (75, 0, 130),      # Indigo
    (238, 130, 238)    # Violet
]

# Function to generate connection color based on connection index
# This loops through the defined CONNECTION_COLORS to provide a dynamic color scheme
def generate_connection_color(index, total_connections):
    color_index = index % len(CONNECTION_COLORS)  # Cycle through the list of colors
    return CONNECTION_COLORS[color_index]  # Return the corresponding color

# Open a video stream (replace with 0 to use the default camera, 1 for additional camera, or path video file (r"Your_PATH"))
cap = cv2.VideoCapture(r"C:\Users\Arsyi Aditama\Downloads\3196221-uhd_3840_2160_25fps.mp4")

# Main processing loop for video frames
while True:
    # Read a frame from the video capture
    ret, img = cap.read()  # `ret` indicates if a frame was successfully captured
    if not ret:  # Break the loop if no frame is captured
        break

    # Resize the captured frame to 800x600 for faster processing
    img_resized = cv2.resize(img, (800, 600))

    # Perform object detection on the resized frame using YOLOv8
    results_yolo = model(img_resized)

    # Create a blank image (same dimensions as `img_resized`) to display pose only
    blank_image = np.ones_like(img_resized) * 0  # Initialize a black image

    # Loop through detected objects (bounding boxes) in the frame
    for result in results_yolo[0].boxes:
        # Extract bounding box coordinates (x1, y1, x2, y2)
        x1, y1, x2, y2 = map(int, result.xyxy[0])  # Convert coordinates to integers

        # Crop the region of interest (ROI) from the frame based on the bounding box
        cropped_img = img_resized[y1:y2, x1:x2]

        # Perform pose estimation on the cropped image
        result_pose = pose.process(cropped_img)

        # Draw pose landmarks and connections on the cropped image
        if result_pose.pose_landmarks:  # Check if landmarks are detected
            for i, connection in enumerate(mp_pose.POSE_CONNECTIONS):  # Loop through all pose connections
                start_idx, end_idx = connection  # Get start and end landmark indices for the connection

                # Get the coordinates of the start and end landmarks
                start_landmark = result_pose.pose_landmarks.landmark[start_idx]
                end_landmark = result_pose.pose_landmarks.landmark[end_idx]

                # Convert normalized landmark coordinates to pixel coordinates
                start_point = (int(start_landmark.x * cropped_img.shape[1]), int(start_landmark.y * cropped_img.shape[0]))
                end_point = (int(end_landmark.x * cropped_img.shape[1]), int(end_landmark.y * cropped_img.shape[0]))

                # Choose a dynamic color for the connection line
                color = generate_connection_color(i, len(mp_pose.POSE_CONNECTIONS))

                # Draw the connection line on the cropped image
                cv2.line(cropped_img, start_point, end_point, color, 4)

                # Also draw the connection line on the blank image (pose-only view)
                cv2.line(blank_image[y1:y2, x1:x2], start_point, end_point, color, 4)

        # Overlay the cropped image with pose onto the main frame
        img_resized[y1:y2, x1:x2] = cropped_img

    # Display the main frame with all detected poses and bounding boxes
    cv2.imshow("Multi-Person Pose Tracking", img_resized)

    # Display the blank image with pose landmarks and connections only
    cv2.imshow("Pose Only (Landmark and Connections)", blank_image)

    # Exit the loop if the 'x' key is pressed
    if cv2.waitKey(30) & 0xFF == ord('x'):
        break

# Release the video capture object to free system resources
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
