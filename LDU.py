import cv2
import streamlit as st
import time
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist

LANDMARKS_PATH = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(LANDMARKS_PATH)

def calculate_EAR(eye):
    y1 = dist.euclidean(eye[1], eye[5])
    y2 = dist.euclidean(eye[2], eye[4])
    x1 = dist.euclidean(eye[0], eye[3])
    EAR = (y1 + y2) / x1
    return EAR


def LDU(message_container, face_placeholder):
    live= False
    cap = cv2.VideoCapture(0)
    blink_detected = False
    start_time = time.time()

    # Add Blink Detection message
    with message_container:
        st.info("Starting Blink Detection... Please blink to verify you are live. 👁️")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cropped_face = frame[y:y + h, x:x + w]

            cropped_face_resized = cv2.resize(cropped_face, (200, 200))
            cropped_face_resized = cv2.cvtColor(cropped_face_resized, cv2.COLOR_BGR2RGB)

            face_placeholder.image(cropped_face_resized, caption="Live Face", use_container_width=False)

            shape = landmark_predictor(gray, face)
            shape = face_utils.shape_to_np(shape)

            lefteye = shape[face_utils.FACIAL_LANDMARKS_IDXS["left_eye"][0]:face_utils.FACIAL_LANDMARKS_IDXS["left_eye"][1]]
            righteye = shape[face_utils.FACIAL_LANDMARKS_IDXS["right_eye"][0]:face_utils.FACIAL_LANDMARKS_IDXS["right_eye"][1]]

            left_EAR = calculate_EAR(lefteye)
            right_EAR = calculate_EAR(righteye)

            if (left_EAR + right_EAR) / 2 < 0.45:
                blink_detected = True
                break

        if time.time() - start_time > 10:
            with message_container:
                st.error("No blink detected! Not a Live Person.")
            break

            

        if blink_detected:
            with message_container:
                st.success("Blink Detected! User is a Live Person.")
                live=True
            break

    cap.release()
    return live
    


