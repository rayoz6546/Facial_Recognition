import cv2
import streamlit as st
import time
import mediapipe as mp
from scipy.spatial import distance as dist

# Mediapipe face mesh setup
mp_face_mesh = mp.solutions.face_mesh

# EAR calculation function
def calculate_EAR(eye):
    y1 = dist.euclidean(eye[1], eye[5])
    y2 = dist.euclidean(eye[2], eye[4])
    x1 = dist.euclidean(eye[0], eye[3])
    EAR = (y1 + y2) / x1
    return EAR

# Mediapipe-based Blink Detection Function
def LDU(message_container, face_placeholder):
    live = False
    cap = cv2.VideoCapture(0)
    blink_detected = False
    start_time = time.time()

    # Add Blink Detection message
    with message_container:
        st.info("Starting Blink Detection... Please blink to verify you are live. 👁️")

    # Mediapipe face mesh solution
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Extracting landmarks for eyes
                    left_eye = [
                        (face_landmarks.landmark[i].x * frame.shape[1],
                         face_landmarks.landmark[i].y * frame.shape[0])
                        for i in [33, 160, 158, 133, 153, 144]
                    ]
                    right_eye = [
                        (face_landmarks.landmark[i].x * frame.shape[1],
                         face_landmarks.landmark[i].y * frame.shape[0])
                        for i in [362, 385, 387, 263, 373, 380]
                    ]

                    # EAR Calculation
                    left_EAR = calculate_EAR(left_eye)
                    right_EAR = calculate_EAR(right_eye)

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
                    live = True
                break

    cap.release()
    return live

    


