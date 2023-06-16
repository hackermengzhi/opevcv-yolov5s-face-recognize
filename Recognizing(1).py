import numpy as np
import cv2

haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

people = ['Ya', 'Ye', 'Weng', 'Jia','Yi']
features = np.load('features.npy', allow_pickle=True)
labels = np.load('labels.npy', allow_pickle=True)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')


# RTSP URL（根据实际情况修改）
#rtsp_url = "http://127.0.0.1:5000/video_feed"
# 创建一个 VideoCapture 对象，参数可以是设备的索引号，或者是一个视频文件
cap = cv2.VideoCapture(0)

while True:
    # 逐帧捕获
    ret, frame = cap.read()

    # 将捕获的帧转换为灰度
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 在帧上检测人脸
    face_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

    face_rect = sorted(face_rect, key=lambda x: x[2] * x[3], reverse=True)

    if len(face_rect) > 0:
        (x, y, w, h) = face_rect[0]  # 获取最大的人脸框
        face_roi = gray[y:y + h, x:x + w]

        label, confidence = face_recognizer.predict(face_roi)
        print(f'标签 = {people[label]}，置信度 = {confidence}')
        if(confidence<=35):
            la=str(people[label])
        else:
            la="Unknown"
        cv2.putText(frame, la, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 显示结果帧
    cv2.imshow('人脸识别', frame)

    # 如果按下 q 键，就跳出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 完成所有操作后，释放 VideoCapture 对象
cap.release()
cv2.destroyAllWindows()
