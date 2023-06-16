import cv2
import fastdeploy.vision as vision

# Load the model
model = vision.detection.PPYOLOE("ppyoloe_crn_l_300e_coco/model.pdmodel",
                                 "ppyoloe_crn_l_300e_coco/model.pdiparams",
                                 "ppyoloe_crn_l_300e_coco/infer_cfg.yml")

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
    vis_frame = vision.vis_detection(frame, result, score_threshold=0.5)

    # Display the frame
    cv2.imshow('Face Detection', vis_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()
