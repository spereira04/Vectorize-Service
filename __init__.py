import cv2
import numpy as np
from PIL import Image
from imgbeddings import imgbeddings
import base64
import io
import json

from flask import Flask, jsonify, request

app = Flask(__name__)

haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +  'haarcascade_frontalface_default.xml')

@app.route('/', methods=['POST'])
def enpoin():
    # data = json.loads(request.data)
    # base64String = data['base64String']
    base64String = request.data

    image = Image.open(io.BytesIO(base64.b64decode(base64String)))
    current_frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    current_faces = haar_cascade.detectMultiScale(current_frame, scaleFactor=1.05, minNeighbors=3, minSize=(300, 300))
    if len(current_faces) < 1:
        return jsonify({'error': 'No face detected'}), 400

    x, y, w, h = current_faces[0]
    image = current_frame[y:y+h, x:x+w]
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    ibed = imgbeddings()
    _, buffer = cv2.imencode('.jpg', gray_image)
    embedding = ibed.to_embeddings(Image.open(io.BytesIO(buffer)))[0] 

    return jsonify({
        'vector': embedding.tolist(), 
        'base64String': base64.b64encode(buffer).decode('utf-8')
    })

if __name__ == '__main__':
    app.run()
