import cv2
import numpy as np
import os

def load_known_faces(directory):
    known_faces = []
    known_names = []

    for filename in os.listdir(directory):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image = cv2.imread(os.path.join(directory, filename))
            face_encoding = encode_face(image)
            known_faces.append(face_encoding)
            known_names.append(os.path.splitext(filename)[0])
    
    return known_faces, known_names

def encode_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        return face.flatten()
    
    return None

def recognize_face(known_faces, known_names, image):
    input_face = encode_face(image)
    if input_face is None:
        return None, None
    
    matches = []
    for known_face in known_faces:
        match = np.linalg.norm(known_face - input_face)
        matches.append(match)
    
    best_match_index = np.argmin(matches)
    if matches[best_match_index] < 1000:  # Threshold value
        return known_names[best_match_index], matches[best_match_index]
    
    return None, None
