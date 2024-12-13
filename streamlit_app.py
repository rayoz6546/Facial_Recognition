import os
import streamlit as st
from PIL import Image
from FRU import FRU
from LDU import LDU
from image_collection import clear_directory, collect_images, collect_live_images, face_extractor
# Paths
IMAGE_FOLDER = "images"
AUTHORIZED_FOLDER = os.path.join(IMAGE_FOLDER, "1")
UNAUTHORIZED_FOLDER = os.path.join(IMAGE_FOLDER, "0")
LIVE_FOLDER = "live_images"
LANDMARKS_PATH = "shape_predictor_68_face_landmarks.dat"
HAAR_CASCADE_PATH = os.path.abspath("haarcascade_frontalface_default.xml")

os.makedirs(AUTHORIZED_FOLDER, exist_ok=True)
os.makedirs(UNAUTHORIZED_FOLDER, exist_ok=True)

# Title
st.set_page_config(page_title="Facial Recognition Dashboard", page_icon="👀", layout="wide")
st.title("👀 Facial Recognition Dashboard")
#----------------------------------------------------------------------------------------------------------#
st.markdown("""
    <style>
    .stApp {  
        background-color: #f0f8ff !important;  /* Light blue background */
    }
    .stButton>button {
        background-color: #ff4b5c;
        color: white;
        font-size: 18px;
        padding: 15px 40px;
        border-radius: 12px;
        border: 2px solid #ff4b5c;
    }
    .stButton>button:hover {
        background-color: #ff1e2a;
    }
    .stText {
        font-family: 'Comic Sans MS', cursive, sans-serif;
        color: #ff5733;
    }
    .stMarkdown {
        font-family: 'Comic Sans MS', cursive, sans-serif;
        color: #ff5733;
        font-size: 20px;
    }

    /* Custom styling for the face window */
    .face-window {
        position: fixed;
        top: 10%;
        right: 10%;

    }
            
    .blink { 
        position: relative;
        
            }

    </style>
""", unsafe_allow_html=True)
#--------------------------------------------------------------------------------------------------------#
def handle_authenticate():
    with messages:
        st.info("Capturing images... Please hold still.")
    collect_images()
    with messages:
        st.success("🎉 You are now an authorized user!")
    st.session_state.authenticated = True

def handle_unlock():
    if not st.session_state.authenticated:
        with messages:
            st.error("❌ Unauthorized User. Please Authenticate Yourself to Become an Authorized User")
    else:
        live = LDU(message_container, face_placeholder)

        if live:
            with message_container:
                st.info("Starting Facial Recognition...")

            ret, frame = collect_live_images(message_container)
            if not ret:
                with message_container:
                    st.error("Failed to capture video.")
            else:
                face = face_extractor(frame)
                if face is not None:

                    verified, training, testing = FRU(AUTHORIZED_FOLDER, UNAUTHORIZED_FOLDER, LIVE_FOLDER)
                    if verified:
                        with message_container:
                            st.success("✅ User Authorized!")
                            st.session_state.lock_status = 'lock_image_unlocked.png'  # Update to unlocked image
                            st.session_state.unlock_enabled = False
                            st.session_state.lock_enabled = True


                    else:
                        with message_container:
                            st.error("❌ Unauthorized User")
                else:
                    with message_container:
                        st.error("No face detected during authentication.")

def handle_lock():
    st.session_state.lock_status = 'lock_image_locked.png'
    st.session_state.unlock_enabled = True
    st.session_state.lock_enabled = False

def handle_reset():
    reset_states()
    clear_directory(AUTHORIZED_FOLDER)
    clear_directory(LIVE_FOLDER)
    st.session_state.authenticated = False
    st.session_state.unlock_enabled = True
    st.session_state.lock_enabled = False
    st.session_state.lock_status = 'lock_image_locked.png'
    with messages:
        st.success("🔄 All data reset. You can start again.")

def reset_states():
    if "unlock_enabled" not in st.session_state:       
        st.session_state.unlock_enabled = True
    if "lock_enabled" not in st.session_state:   
        st.session_state.lock_enabled = False

    # st.session_state.authenticated = False
    # st.session_state.lock_status = 'lock_image_locked.png' 

    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
        clear_directory(AUTHORIZED_FOLDER)
        clear_directory(LIVE_FOLDER)

    if 'lock_status' not in st.session_state:
        st.session_state.lock_status = 'lock_image_locked.png' 

reset_states()


# Create buttons on the same line
col1, col2, col3, col4, col5 = st.columns([1, 2, 1, 1, 1]) 

with col1: 
    lock_image = Image.open(st.session_state.lock_status)
    lock_image_placeholder = st.image(lock_image, width=100, caption="Lock Status")

with col2:
    authenticate_button = st.button("💼 Authenticate User", key="authenticate",disabled=st.session_state.authenticated, on_click=handle_authenticate)

with col3:
    unlock_button = st.button("🔑 Unlock", disabled=not st.session_state.unlock_enabled,on_click=handle_unlock)

with col4:
    lock_button = st.button("🔒 Lock", disabled = not st.session_state.lock_enabled, on_click=handle_lock)
with col5:
    reset_button = st.button("Reset", on_click=handle_reset)




messages = st.container()


          
col1, col2 = st.columns([2, 1]) 
with col1:
    face_placeholder = st.empty()  
with col2:
    message_container = st.container()  




