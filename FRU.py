import numpy as np
import os 
from PIL import Image
import joblib 
from svm import train

def load_images(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):
            img = Image.open(os.path.join(folder_path, filename)).convert('L')  
            img_array = np.array(img.resize((100, 100))).flatten()  
            images.append(img_array)
    return np.array(images)

def FRU(authorized_path, unauthorized_path, live_folder):
    
    current_path = os.path.abspath("images/")
    # d = os.path.join(current_path, "images")
    authorized_path = os.path.join(current_path, "1")  # Authorized users
    unauthorized_path = os.path.join(current_path, "0")  # Unauthorized users

    # Ensure directories exist
    os.makedirs(authorized_path, exist_ok=True)
    os.makedirs(unauthorized_path, exist_ok=True)
    input_dir = authorized_path
    verified = False

    training_accuracy, testing_accuracy = train()
    
    recognizer= joblib.load('svm_model.joblib')
    model_filename = 'svm_model.joblib'
    pca_filename = 'pca_transform.joblib'
    scaler_filename = 'scaler_transform.joblib'
    
    recognizer = joblib.load(model_filename)
    pca = joblib.load(pca_filename)
    scaler = joblib.load(scaler_filename)




    X_live = []
    images_live = load_images(live_folder)
    X_live.extend(images_live)
    # X_live = np.array(X_live)

    # if X_live.size == 0:
    #     print("No live images found for prediction!")
    #     return False

    # Apply preprocessing (scaling and PCA)
    X_live_scaled = scaler.transform(X_live)
    X_live_pca = pca.transform(X_live_scaled)

    y_predict = recognizer.predict(X_live_pca)


    majority = np.bincount(y_predict).argmax()

    if majority == 1:
        print("VERIFIED")
        verified=True
    else:
        print("UNVERIFIED")
        verified = False
    
    return verified, training_accuracy, testing_accuracy

