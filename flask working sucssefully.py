from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import face_recognition
import os
import base64  # Import base64 module for encoding images
import numpy as np

app = Flask(__name__)
CORS(app)

known_faces = []  # Store known face encodings
image_folder = 'images'  # Folder containing known face images

for image_file in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_file)
    image = face_recognition.load_image_file(image_path)
    face_encoding = face_recognition.face_encodings(image)[0]
    known_faces.append(face_encoding)

@app.route('/verify', methods=['POST'])
def verify_image():
    try:
        uploaded_image = request.files['image'].read()

        pil_image = Image.open(io.BytesIO(uploaded_image))

        uploaded_face_encodings = face_recognition.face_encodings(np.array(pil_image))

        if len(uploaded_face_encodings) == 0:
            return jsonify({'message': 'No face found in the uploaded image.'}), 400

        uploaded_face_encoding = uploaded_face_encodings[0]

        matches = []
        for known_face_encoding in known_faces:
            match = face_recognition.compare_faces([known_face_encoding], uploaded_face_encoding)[0]
            if match:
                # Convert the matching image to base64 and append to the matches list
                with open(image_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode("utf-8")
                    matches.append(base64_image)

        return jsonify({'message': 'Matching faces found.', 'matches': matches})
    except Exception as e:
        return jsonify({'message': 'Error: {}'.format(str(e))}), 500

if __name__ == '__main__':
    app.run()
