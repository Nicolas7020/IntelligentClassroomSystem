import json
import requests
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import os
from config import headers

# Run inference on an image
url = "https://api.ultralytics.com/v1/predict/R6nMlK6kQjSsQ76MPqQM"
data = {"size": 640, "confidence": 0.25, "iou": 0.45}
with open("data.jpg", "rb") as f:
    response = requests.post(url, headers=headers, data=data, files={"image": f})

# Check for successful response
response.raise_for_status()

data = response.json()

print(data)

objects = data['images'][0]['results']

# Open picture
image = Image.open('data.jpg').convert('RGB')
image_backups = image
draw = ImageDraw.Draw(image)

# Label the results on the picture and cut the picture
for i, obj in enumerate(objects):
    if obj["name"] != "person":
        continue
    box = obj['box']
    cropped_image = image_backups.crop((box['x1'], box['y1'], box['x2'], box['y2']))
    plt.imshow(cropped_image)
    plt.show()
    name = input("Please enter this person's name:")
    if name == "sb":
        continue
    if not os.path.exists(f'./face_data/{name}/'):
        os.mkdir(f'./face_data/{name}/')
        cropped_image_rgb = cropped_image.convert('RGB')
        cropped_image.save(f'./face_data/{name}/1.jpg')


print("The image has been successfully cut and saved locally, and the result is marked on the original image.")
