import json
import requests
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import os

# Run inference on an image
url = "https://api.ultralytics.com/v1/predict/vY89GpzxNSze7BxrYy72"
headers = {"x-api-key": "daa17fbe445f42dbab18a57970faa17ae0c0ea5947"}
data = {"size": 640, "confidence": 0.25, "iou": 0.45}
with open("nb.jpg", "rb") as f:
    response = requests.post(url, headers=headers, data=data, files={"image": f})

# Check for successful response
response.raise_for_status()

data = response.json()

objects = data['data']

# Open camera
image = Image.open('nb.jpg')
image_backups = image
draw = ImageDraw.Draw(image)

# Label the results on the picture and cut the picture
for i, obj in enumerate(objects):
    box = obj['box']
    cropped_image = image_backups.crop((box['x1'], box['y1'], box['x2'], box['y2']))
    cropped_image.save(f'./img/cropped_{i}.jpg')
    draw.rectangle([box['x1'], box['y1'], box['x2'], box['y2']], outline='red')
    draw.text((box['x1'], box['y1'] - 10), obj['name'], fill='red')

# Show picture
plt.imshow(image)
plt.show()

print("The image has been successfully cut and saved locally, and the result is marked on the original image.")
