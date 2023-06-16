import cv2
import fastdeploy.vision as vision

# Load the model
model = vision.facedet.YOLOv5Face("yolov5s-face.onnx")

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Predict the face detection result
    result = model.predict(frame)

    # Visualize the prediction result
    vis_frame = vision.vis_face_detection(frame, result)

    # Display the frame
    cv2.imshow('Face Detection', vis_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()
