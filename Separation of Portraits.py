import requests
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# Run inference on an image
url = "https://api.ultralytics.com/v1/predict/R6nMlK6kQjSsQ76MPqQM"
headers = {"x-api-key": "97f6418132004515aff7e79678c21f4aa65c004bb4"}
data = {"size": 640, "confidence": 0.25, "iou": 0.45}
with open("data.jpg", "rb") as f:
    response = requests.post(url, headers=headers, data=data, files={"image": f})

# Check for successful response
response.raise_for_status()

data = response.json()

objects = data['data']

# Open picture
image = Image.open('data.jpg')
draw = ImageDraw.Draw(image)

# Label the results on the picture and cut the picture
for i, obj in enumerate(objects):
    if obj["name"] != "person":
        continue
    box = obj['box']
    draw.rectangle((box['x1'], box['y1'], box['x2'], box['y2']), outline="red",width=2)

plt.imshow(image)
plt.show()
