from flask import Flask, render_template, request, redirect, url_for
import cv2
import os
from face_recognition import load_known_faces, recognize_face

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
KNOWN_FACES_FOLDER = 'known_faces'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

known_faces, known_names = load_known_faces(KNOWN_FACES_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        image = cv2.imread(filepath)

        name, _ = recognize_face(known_faces, known_names, image)
        if name:
            return render_template('result.html', name=name)
        else:
            return render_template('result.html', name='Unknown')
    
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
