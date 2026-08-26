import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import pytesseract
from sklearn.metrics.pairwise import cosine_similarity
import time  # Add this import at the top of the file
from PIL import Image
import re
import threading
import os
import json
import pickle

# Add this line before using pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Adjust the path as needed

# Replace the face model and related functions with HOG-based approach
def compute_hog(image):
    resized_image = cv2.resize(image, (128, 128))
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    cell_size = (8, 8)
    block_size = (2, 2)
    nbins = 9
    
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=1)
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    angle_bins = np.int32(nbins * angle / 360)
    
    h, w = gray.shape
    hist = np.zeros((h // cell_size[0], w // cell_size[1], nbins), dtype=np.float32)
    for i in range(hist.shape[0]):
        for j in range(hist.shape[1]):
            for k in range(cell_size[0]):
                for l in range(cell_size[1]):
                    y, x = i * cell_size[0] + k, j * cell_size[1] + l
                    bin_idx = angle_bins[y, x]
                    hist[i, j, bin_idx] += mag[y, x]
    
    hog_descriptor = []
    for i in range(hist.shape[0] - block_size[0] + 1):
        for j in range(hist.shape[1] - block_size[1] + 1):
            block_hist = hist[i:i + block_size[0], j:j + block_size[1]].ravel()
            block_hist /= np.linalg.norm(block_hist) + 1e-7
            hog_descriptor.extend(block_hist)
    
    return np.array(hog_descriptor)

# Replace calculate_similarity with this function
def compute_similarity_cosine(hog1, hog2):
    dot_product = np.dot(hog1, hog2)
    norm_a = np.linalg.norm(hog1)
    norm_b = np.linalg.norm(hog2)
    similarity = dot_product / (norm_a * norm_b + 1e-7)
    return similarity

# Function to preprocess image for face recognition
def preprocess_face(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        face = image[y:y+h, x:x+w]
        face = cv2.resize(face, (100, 100))
        return face
    else:
        print("No face detected")
        return None

# Function to preprocess license plate
def preprocess_license_plate(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    # Find contours and select the largest one
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros(thresh.shape, np.uint8)
        cv2.drawContours(mask, [largest_contour], 0, 255, -1)
        result = cv2.bitwise_and(thresh, thresh, mask=mask)
    else:
        result = thresh
    
    return result

# Add this function to detect and crop license plate region
def detect_license_plate(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY_INV, 19, 2)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Initialize variables
    plate_img = None
    max_area = 0
    
    # Filter contours based on area and aspect ratio
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h
        area = w * h
        
        # Filter based on aspect ratio and minimum area
        if 1.5 < aspect_ratio < 6 and area > 5000:
            if area > max_area:
                max_area = area
                # Add padding
                pad_x = int(w * 0.1)
                pad_y = int(h * 0.1)
                
                # Ensure coordinates are within image bounds
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(image.shape[1], x + w + pad_x)
                y2 = min(image.shape[0], y + h + pad_y)
                
                plate_img = image[y1:y2, x1:x2]
    
    if plate_img is not None:
        # Show the detected region (for debugging)
        cv2.imshow("Detected License Plate", plate_img)
        cv2.waitKey(1000)
        return plate_img
    
    print("No license plate detected")
    return None

# Update the recognize_license_plate function
def recognize_license_plate(image):
    # First detect and crop the license plate
    plate_img = detect_license_plate(image)
    if plate_img is None:
        return None
    
    # Preprocess the cropped plate
    processed = preprocess_license_plate(plate_img)
    processed = cv2.resize(processed, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    configs = [
        '--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        '--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        '--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        '--oem 3 --psm 13 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
    ]
    
    for config in configs:
        text = pytesseract.image_to_string(processed, config=config)
        clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
        
        # Match the pattern: 1-2 letters, 1-4 numbers, 1-2 letters
        match = re.search(r'([A-Z]{1,2})(\d{1,4})([A-Z]{1,2})', clean_text)
        if match:
            plate = ' '.join(match.groups())
            
            # Post-processing to ensure correct format
            parts = plate.split()
            if len(parts) == 3 and len(parts[0]) == 1 and len(parts[1]) <= 4 and len(parts[2]) == 2:
                final_plate = f"{parts[0]} {parts[1]} {parts[2]}"
                print(f"Detected license plate: {final_plate}")
                cv2.imshow("Processed License Plate", processed)
                cv2.waitKey(1000)
                cv2.destroyAllWindows()
                return final_plate
    
    print("No valid license plate detected")
    cv2.imshow("Processed License Plate", processed)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
    return None

# Add this function to visualize face detection
def show_detected_face(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    cv2.imshow("Detected Face", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Add this function after the preprocess_license_plate function
def save_preprocessed_image(image, image_type, status):
    """
    Save preprocessed image with timestamp
    
    Args:
        image: The preprocessed image to save
        image_type: String indicating 'face' or 'license'
        status: String indicating 'entry' or 'exit'
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"preprocessed_{image_type}{status}{timestamp}.jpg"
    cv2.imwrite(filename, image)
    print(f"Saved {image_type} image as: {filename}")

# Main parking system class
class ParkingSystem:
    def __init__(self):
        self.entries = {}
        self.data_dir = "parking_data"
        self.load_data()  # Load saved data when initializing
        
        # Create data directory if it doesn't exist
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            os.makedirs(os.path.join(self.data_dir, "faces"))
            os.makedirs(os.path.join(self.data_dir, "plates"))
    
    def save_data(self):
        """Save parking entries data to file"""
        data_file = os.path.join(self.data_dir, "entries.pkl")
        with open(data_file, 'wb') as f:
            pickle.dump(self.entries, f)
    
    def load_data(self):
        """Load parking entries data from file"""
        data_file = os.path.join(self.data_dir, "entries.pkl")
        if os.path.exists(data_file):
            with open(data_file, 'rb') as f:
                self.entries = pickle.load(f)
    
    def search_records(self, license_plate=None):
        """Search for parking records"""
        if license_plate:
            if license_plate in self.entries:
                return f"Found record for license plate: {license_plate}"
            return "No record found for this license plate"
        else:
            return f"Current entries: {list(self.entries.keys())}"

    def vehicle_entry(self, face_image, license_image):
        # Process face image
        show_detected_face(face_image.copy())
        face = preprocess_face(face_image)
        if face is not None:
            save_preprocessed_image(face, "face", "entry")
        face_features = compute_hog(face_image)

        # Process license plate image separately
        plate_img = detect_license_plate(license_image)
        if plate_img is not None:
            processed_plate = preprocess_license_plate(plate_img)
            save_preprocessed_image(processed_plate, "license", "entry")
        license_plate = recognize_license_plate(license_image)
        
        if face_features is not None and license_plate:
            self.entries[license_plate] = face_features
            return f"Vehicle with license plate {license_plate} entered."
        elif face_features is None:
            return "Entry failed. Could not recognize face."
        elif license_plate is None:
            return "Entry failed. Could not recognize license plate."
        else:
            return "Entry failed. Could not recognize face or license plate."

    def vehicle_exit(self, face_image, license_image):
        # Process face image
        show_detected_face(face_image.copy())
        face = preprocess_face(face_image)
        if face is not None:
            save_preprocessed_image(face, "face", "exit")
        face_features = compute_hog(face_image)

        # Process license plate image separately
        plate_img = detect_license_plate(license_image)
        if plate_img is not None:
            processed_plate = preprocess_license_plate(plate_img)
            save_preprocessed_image(processed_plate, "license", "exit")
        license_plate = recognize_license_plate(license_image)
        
        if face_features is not None and license_plate:
            if license_plate in self.entries:
                similarity = compute_similarity_cosine(face_features, self.entries[license_plate])
                if similarity > 0.7:
                    del self.entries[license_plate]
                    return f"Vehicle with license plate {license_plate} exited. Face similarity: {similarity:.2f}"
                else:
                    return f"Exit denied. Face does not match entry data. Similarity: {similarity:.2f}"
            else:
                return "Exit denied. License plate not found in entry records."
        elif face_features is None:
            return "Exit failed. Could not recognize face."
        elif license_plate is None:
            return "Exit failed. Could not recognize license plate."
        else:
            return "Exit failed. Could not recognize face or license plate."

def capture_images():
    webcam = cv2.VideoCapture(0)  # Webcam for face
    builtin_cam = cv2.VideoCapture(1)  # Built-in camera for license plate

    if not webcam.isOpened() or not builtin_cam.isOpened():
        print("Error: Cannot open one or both cameras")
        return None, None

    print("Preparing to capture images in 5 seconds...")
    start_time = time.time()

    webcam_frame = None
    builtin_frame = None

    def capture_webcam():
        nonlocal webcam_frame
        ret, webcam_frame = webcam.read()

    def capture_builtin():
        nonlocal builtin_frame
        ret, builtin_frame = builtin_cam.read()

    while True:
        webcam_thread = threading.Thread(target=capture_webcam)
        builtin_thread = threading.Thread(target=capture_builtin)

        webcam_thread.start()
        builtin_thread.start()

        webcam_thread.join()
        builtin_thread.join()

        if webcam_frame is None or builtin_frame is None:
            print("Error: Cannot read frames from one or both cameras")
            webcam.release()
            builtin_cam.release()
            return None, None

        elapsed_time = int(time.time() - start_time)
        remaining_time = 5 - elapsed_time

        if remaining_time <= 0:
            break

        # Add countdown text to the frames
        cv2.putText(webcam_frame, f"Capturing in: {remaining_time}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(builtin_frame, f"Capturing in: {remaining_time}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Webcam (Face)", webcam_frame)
        cv2.imshow("Built-in Camera (License Plate)", builtin_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            webcam.release()
            builtin_cam.release()
            cv2.destroyAllWindows()
            return None, None

    print("Capturing images now!")
    ret_webcam, final_webcam_frame = webcam.read()
    ret_builtin, final_builtin_frame = builtin_cam.read()

    webcam.release()
    builtin_cam.release()
    cv2.destroyAllWindows()

    if not ret_webcam or not ret_builtin:
        print("Error: Cannot capture images from one or both cameras")
        return None, None

    return final_webcam_frame, final_builtin_frame  # Return face image first, then license plate image

# Add this function for the menu system
def display_menu():
    print("\n=== Parking System Menu ===")
    print("1. Vehicle Entry")
    print("2. Vehicle Exit")
    print("3. Search Records")
    print("4. Exit Program")
    print("========================")
    return input("Choose an option (1-4): ")

# Modify the main execution part
def main():
    parking_system = ParkingSystem()
    
    while True:
        choice = display_menu()
        
        if choice == '1':
            print("\nInitiating vehicle entry process...")
            entry_face_image, entry_license_image = capture_images()
            
            if entry_face_image is not None and entry_license_image is not None:
                cv2.imshow("Entry Face Image", entry_face_image)
                cv2.imshow("Entry License Plate Image", entry_license_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
                print(parking_system.vehicle_entry(entry_face_image, entry_license_image))
            else:
                print("Failed to capture entry images")
                
        elif choice == '2':
            print("\nInitiating vehicle exit process...")
            exit_face_image, exit_license_image = capture_images()
            
            if exit_face_image is not None and exit_license_image is not None:
                cv2.imshow("Exit Face Image", exit_face_image)
                cv2.imshow("Exit License Plate Image", exit_license_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
                print(parking_system.vehicle_exit(exit_face_image, exit_license_image))
            else:
                print("Failed to capture exit images")
                
        elif choice == '3':
            search_option = input("\nEnter license plate to search (or press Enter to list all): ")
            print(parking_system.search_records(search_option))
                
        elif choice == '4':
            print("\nThank you for using the Parking System!")
            break
            
        else:
            print("\nInvalid option! Please choose 1-4.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()