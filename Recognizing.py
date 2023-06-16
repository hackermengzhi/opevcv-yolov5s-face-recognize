import numpy as np
import cv2
import fastdeploy.vision as vision

# Load the YOLOv5 face detection model
face_detection_model = vision.facedet.YOLOv5Face("yolov5s-face.onnx")

# Load the face recognition model
face_recognition_model = cv2.face.LBPHFaceRecognizer_create()
face_recognition_model.read('face_trained.yml')

# Load the face detection cascade classifier
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

people = ['Ya', 'Ye', 'Weng', 'Jia']
features = np.load('features.npy', allow_pickle=True)
labels = np.load('labels.npy', allow_pickle=True)

# RTSP URL（根据实际情况修改）
rtsp_url = "http://127.0.0.1:5000/video_feed"

# 创建一个 VideoCapture 对象，参数可以是设备的索引号，或者是一个视频文件
cap = cv2.VideoCapture(rtsp_url)

while True:
    # 逐帧捕获
    ret, frame = cap.read()

    if not ret:
        break

    # 使用YOLOv5模型进行人脸检测
    detection_result = face_detection_model.predict(frame)
    # print(detection_result)
    # # 获取人脸框信息
    # xmin, ymin, xmax, ymax = detection_result.xmin,detection_result.ymin,detection_result.xmax,detection_result.ymax
    boxes = detection_result.boxes
    for box in boxes:
        xmin, ymin, xmax, ymax = box
    # 提取人脸框的坐标

    w = (xmax - xmin)*1.1
    h = (ymax - ymin)*1.1
    x = xmin-30
    y = ymin-30
    # 提取人脸ROI
    face_roi = frame[int(ymin):int(ymax), int(xmin):int(xmax)]

    # 将人脸ROI转换为灰度图像
    gray_face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

    # 在灰度人脸ROI上使用Haar级联分类器进行人脸检测
    face_rect_haar = haar_cascade.detectMultiScale(gray_face_roi, 1.1, 4)

    if len(face_rect_haar) > 0:
            (fx, fy, fw, fh) = face_rect_haar[0]
            face_roi_haar = gray_face_roi[fy:fy + fh, fx:fx + fw]

            # 使用人脸识别模型进行人脸识别
            label, confidence = face_recognition_model.predict(face_roi_haar)
            print(f'标签 = {people[label]}，置信度 = {1/confidence}')

            if confidence <= 200:
                la = str(people[label])
            else:
                la = "Unknown"

            # 在原始帧上绘制人脸框和标签
            cv2.putText(frame, la, (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)

    # 显示结果帧
    cv2.imshow('人脸识别', frame)

    # 如果按下 q 键，就跳出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 完成所有操作后，释放 VideoCapture 对象
cap.release()
cv2.destroyAllWindows()
