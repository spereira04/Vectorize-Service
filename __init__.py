import cv2
import numpy as np
from PIL import Image
from imgbeddings import imgbeddings
import base64
import io

from flask import Flask, jsonify, request

app = Flask(__name__)

haar_cascade = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")

@app.route('/', methods=['POST'])
def enpoin():

    data = request.files['file']

    # image_data = base64.b64decode(data)
    image = Image.open(io.BytesIO(bytearray(data.read())))

    current_frame = np.array(image)

    current_faces = haar_cascade.detectMultiScale(current_frame, scaleFactor=1.05, minNeighbors=3, minSize=(300, 300))

    if len(current_faces) < 1:
        exit()
    else:
        x, y, w, h = current_faces[0]

    image = current_frame[y:y+h, x:x+w]

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('resources/gray-face.jpg', gray_image)

    ibed = imgbeddings()
    embedding = ibed.to_embeddings('resources/gray-face.jpg')[0]

    return jsonify({'vector': embedding.tolist(), 'image_bytes': gray_image.tolist()})

if __name__ == '__main__':
    app.run()
