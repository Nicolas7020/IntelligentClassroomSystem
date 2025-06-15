import face_recognition
import os
import time
import tqdm
import cv2

tags_to_names = {0: "drink", 1: "listen", 2: "trance", 3: "write"}
class_names = ['drink', 'listen', 'trance', 'write']
url = "https://predict.ultralytics.com"
headers = {"x-api-key": "97f6418132004515aff7e79678c21f4aa65c004bb4"}
# daa17fbe445f42dbab18a57970faa17ae0c0ea5947
data = {"model": "https://hub.ultralytics.com/models/R6nMlK6kQjSsQ76MPqQM", "imgsz": 640, "conf": 0.25, "iou": 0.45}
font = cv2.FONT_HERSHEY_SIMPLEX

# get"./data"All directory names under the directory
directories = [d for d in os.listdir('./face_data/') if os.path.isdir(os.path.join('./face_data/', d))]

print(directories)
time.sleep(0.1)

face_encodings = {}
face_encodings_list = []
face_names = []


def get_name(path, name):
    image = face_recognition.load_image_file(path)
    list_of_face_encodings = face_recognition.face_encodings(image)
    try:
        face_encodings[name] = list_of_face_encodings[0]
        face_encodings_list.append(list_of_face_encodings[0])
        face_names.append(name)
    except:
        print(name, "This person has no face")


pbar = tqdm.tqdm(total=35, unit="person")

print("Loading face data")
for directory in directories:
    time_od = time.time()
    file_path = os.path.join('./face_data', directory, '1.jpg')
    get_name(file_path, directory)
    pbar.set_description("Create a code for a face")
    pbar.update(1)
    pbar.set_postfix({"name": f"{directory} Code created successfully", "time consuming": f"{time.time() - time_od} MS"})

pbar.close()
print("Face data loading completed！")
