# Directory structure and project outline provided. Below is the implementation.
from tokenize import String

from config import *
import time
import cv2
import tensorflow as tf
import numpy as np
import requests
import sys
from PIL import Image
import os
import face_recognition
import xlwings as xw
import atexit
import keras

# main.py
from PySide6.QtWidgets import QApplication
import sys
import threading


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

# ui/dashboard.py
from PySide6.QtWidgets import *
from PySide6.QtCore import QRect, QCoreApplication, Qt, QThread, Signal, QObject, QRunnable, Slot, QThreadPool
from PySide6.QtGui import QFont, QPixmap, QImage

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Classroom Behavior Detection")
        self.setGeometry(100, 100, 1200, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()



        self.video_label = QLabel("Live Feed")
        self.video_label.setFixedSize(800, 600)
        self.video_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.video_label)

        self.stats_label = QLabel("Class Statistics")
        self.stats_label.setAlignment(Qt.AlignLeft)
        layout.addWidget(self.stats_label)

        self.log_list = QListWidget()
        self.log_list.addItem("Behavior logs will appear here.")
        layout.addWidget(self.log_list)

        self.threadpool = QThreadPool()

        self.start_button = QPushButton("Start Detection")
        self.stop_button = QPushButton("Stop Detection")
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)

        central_widget.setLayout(layout)

        self.start_button.clicked.connect(self.start_detection)
        self.stop_button.clicked.connect(self.stop_detection)

    def start_detection(self):
        self.log_list.addItem("Detection started.")
        self.ai_worker = AIWorker()
        self.ai_worker.signals.imageFeed.connect(self.setVideoFeed)
        self.ai_worker.signals.logSignal.connect(self.addLog)
        self.threadpool.start(self.ai_worker)

    def setVideoFeed(self, image):
        self.video_label.setPixmap(QPixmap.fromImage(image))

    def addLog(self, log):
        self.log_list.addItem(log)

    def stop_detection(self):
        self.log_list.addItem("Detection stopped.")
        self.ai_worker.stop()


class AIWorkerSignals(QObject):
    imageFeed = Signal(QImage)
    logSignal = Signal(str)

class AIWorker(QRunnable):
    def __init__(self):
        super(AIWorker, self).__init__()

        print("AI Initializing")

        self.signals = AIWorkerSignals()
        self.thread_active = True

    def log(self, log):
        self.signals.logSignal.emit(log)

    @Slot()
    def run(self):
        self.signals.logSignal.emit("Initializing table")
        app = xw.App(visible=True, add_book=False)
        wb = app.books.open(r'./test.xlsx')

        for i in range(len(face_names)):
            wb.sheets['sheet1'].range(f'A{i + 1}').value = face_names[i]

        # Line number
        letters = []
        for i in range(26):
            letters.append(chr(ord("A") + i))

        self.log("Initializing variables")

        file_name = "./test_data/nb.jpg"

        IS_DRR = True
        IS_SaveIMG = False

        self.log("Grabbing video stream")
        cap = cv2.VideoCapture(0)
        self.log("success")
        # Load model
        self.log("Load CNN model")
        model = keras.models.load_model('./model/model2.0.keras')
        self.log("Start execution--")

        def image_segmentation(path):
            with open(path, "rb") as f:
                response = requests.post(url, headers=headers, data=data, files={"image": f})

            # Check for successful response
            response.raise_for_status()
            return response.json()

        def gesture_recognition(path):
            img = keras.utils.load_img(
                path, target_size=(112, 112)
            )
            img_array = keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)  # Create a batch

            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0])
            return class_names[np.argmax(score)], 100 * np.max(score)


        x = 2
        while self.thread_active:
            time_global = time.time()
            for framei in range(30):
                ret, frame = cap.read()
            # cv2.imshow("A video", frame)
            cv2.imwrite(file_name, frame)

            image = Image.open(file_name)
            image_backups = image

            res = image_segmentation(file_name)
            self.log(res)
            objects = res['images'][0]['results']

            for i, obj in enumerate(objects):

                t = time.time()
                box = obj['box']
                if obj['name'] != 'person':
                    continue
                cropped_image = image_backups.crop((box['x1'], box['y1'], box['x2'], box['y2']))
                cropped_image.save(f'./img/cropped_{i}.jpg')

                class_name, confidence = gesture_recognition(f'./img/cropped_{i}.jpg')

                picture = face_recognition.load_image_file(f'./img/cropped_{i}.jpg')
                try:
                    encoding = face_recognition.face_encodings(picture)[0]
                except IndexError as e:
                    self.log("no face")
                cv2.putText(frame,
                            class_name,
                            (int(box['x1']), int(box['y1']) + 30),
                            font, 1, (255, 255, 255), 1)

                try:
                    res = face_recognition.face_distance(face_encodings_list, encoding)
                    face_name = face_names[np.argmin(res)]
                    self.log(
                        f"The {i}th portrait in the picture: name={face_name}, behavior={class_name}, behavior confidence={confidence}, portrait processing time:{time.time() - t}")
                    face_index = face_names.index(face_name)
                    wb.sheets['sheet1'].range(f'{letters[x - 1]}{face_index + 1}').value = class_name
                    if IS_DRR:
                        cv2.putText(frame,
                                    face_name,
                                    (int(box['x1']), int(box['y1']) + 60),
                                    font, 1, (255, 255, 255), 1)

                except Exception as e:
                    pass

                os.remove(f'./img/cropped_{i}.jpg')
                cv2.rectangle(frame, (int(box['x1']), int(box['y1'])), (int(box['x2']), int(box['y2'])),
                              (255, 255, 255), 1)
            x += 1
            img_test1 = cv2.resize(frame, (int(frame.shape[0] / 2), int(frame.shape[1] / 2)))
            cv2.imwrite(f"./debug/test{x}.jpg", frame)
            # cv2.imshow("A video", frame)
            print(time_global - time.time())
            c = cv2.waitKey(1)
            convert_to_qt = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format.Format_BGR888)
            pic = convert_to_qt.scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio)
            self.signals.imageFeed.emit(pic)
            if c == 27:
                break

        cv2.destroyAllWindows()

    def stop(self):
        self.thread_active = False
        self.quit()


main()

# resources/styles.qss
"""
QLabel {
    font-size: 14px;
    color: #333;
}
QPushButton {
    background-color: #0078D7;
    color: #fff;
    border-radius: 5px;
    padding: 5px 10px;
}
"""

# ui/main_window.ui
# This file is designed using Qt Designer and converted to a Python file using PySide6-uic.
