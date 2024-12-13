import os
import cv2
import streamlit as st

IMAGE_FOLDER = "images"
AUTHORIZED_FOLDER = os.path.join(IMAGE_FOLDER, "1")
LIVE_FOLDER = "live_images"

HAAR_CASCADE_PATH = os.path.abspath("haarcascade_frontalface_default.xml")
face_classifier = cv2.CascadeClassifier(HAAR_CASCADE_PATH)

def clear_directory(directory):
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            os.remove(file_path)


def face_extractor(img):
    faces = face_classifier.detectMultiScale(img, 1.3, 5)
    if len(faces) == 0:
        return None
    for (x, y, w, h) in faces:
        cropped_face = img[y:y + h, x:x + w]
    return cropped_face


def collect_images():
    cap = cv2.VideoCapture(0)
    count = 0
    
    while count < 100:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video.")
            break
        face = face_extractor(frame)
        if face is not None:
            face = cv2.resize(face, (400, 400))
            filename = os.path.join(AUTHORIZED_FOLDER, f"1_{count}.jpg")
            cv2.imwrite(filename, face)
            count += 1
    cap.release()



def collect_live_images(message_container):
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    count = 0

    while count < 100:
        ret, frame = cap.read()
        if not ret:
            with message_container:
                st.error("Failed to capture video.")
            break

        face = face_extractor(frame)
        if face is not None:
            face = cv2.resize(face, (400, 400))
            filename = os.path.join(LIVE_FOLDER, f"{count}.jpg")
            cv2.imwrite(filename, face)
            count += 1
    cap.release()

    return ret, frame

