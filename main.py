import cv2
import mediapipe as mp
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from tkinter import Tk, Label, Frame, Button, StringVar, Toplevel
from tkinter.ttk import Progressbar, Style
import threading

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Function to extract landmarks from hand images using Mediapipe
def extract_landmarks(image):
    with mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.7) as hands:
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                return landmarks
    return None

# Function to load the dataset
def load_dataset(dataset_path):
    data = []
    labels = []
    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)
        if os.path.isdir(label_path):
            for image_file in os.listdir(label_path):
                image_path = os.path.join(label_path, image_file)
                image = cv2.imread(image_path)
                if image is not None:
                    image = cv2.resize(image, (224, 224))  # Resize image
                    landmarks = extract_landmarks(image)
                    if landmarks:
                        data.append(landmarks)
                        labels.append(label)
    return np.array(data), np.array(labels)

# Load the dataset
dataset_path = r"sign_language_dataset"
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset path '{dataset_path}' does not exist.")
data, labels = load_dataset(dataset_path)

if len(data) == 0 or len(labels) == 0:
    raise ValueError("Dataset is empty. Please check the dataset path and contents.")

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Initialize Tkinter GUI
root = Tk()
root.title("Sign Language Recognition")
root.geometry("800x500")
root.configure(bg="#f5f5f5")

# Variables to store UI state
current_sentence = StringVar(value="")
detected_letter = StringVar(value="Waiting for gesture...")
confidence_level = StringVar(value="Confidence: --")
predicted_letter = ""  # Initialize the global variable

# GUI Layout
top_frame = Frame(root, bg="#f5f5f5")
top_frame.pack(pady=10)

label = Label(top_frame, textvariable=detected_letter, font=("Helvetica", 20, "bold"), bg="#f5f5f5", fg="#333")
label.pack()

confidence_label = Label(top_frame, textvariable=confidence_level, font=("Helvetica", 14), bg="#f5f5f5", fg="#555")
confidence_label.pack(pady=5)

progress = Progressbar(top_frame, orient="horizontal", length=300, mode="determinate", style="green.Horizontal.TProgressbar")
progress.pack(pady=10)

sentence_frame = Frame(root, bg="#f5f5f5")
sentence_frame.pack(pady=20)

sentence_label = Label(sentence_frame, text="Sentence: ", font=("Helvetica", 16), bg="#f5f5f5", fg="#333")
sentence_label.grid(row=0, column=0, sticky="w")

sentence_display = Label(sentence_frame, textvariable=current_sentence, font=("Helvetica", 16), bg="#f5f5f5", fg="#555", wraplength=700, justify="left")
sentence_display.grid(row=1, column=0, columnspan=2, pady=5)

button_frame = Frame(root, bg="#f5f5f5")
button_frame.pack(pady=20)

def update_sentence(is_correct):
    global predicted_letter
    if is_correct:
        new_sentence = current_sentence.get() + predicted_letter
        current_sentence.set(new_sentence)

def clear_sentence():
    current_sentence.set("")

def add_space():
    current_sentence.set(current_sentence.get() + " ")

def start_new_word():
    current_sentence.set(current_sentence.get() + " ")

def undo_last_word():
    sentence = current_sentence.get().strip().split(" ")
    if sentence:
        current_sentence.set(" ".join(sentence[:-1]) + " ")

confirm_button = Button(button_frame, text="Yes", width=10, font=("Helvetica", 12), bg="#4caf50", fg="white", command=lambda: update_sentence(True))
confirm_button.grid(row=0, column=0, padx=10)

reject_button = Button(button_frame, text="No", width=10, font=("Helvetica", 12), bg="#f44336", fg="white", command=lambda: update_sentence(False))
reject_button.grid(row=0, column=1, padx=10)

clear_button = Button(button_frame, text="Clear Sentence", width=15, font=("Helvetica", 12), bg="#2196f3", fg="white", command=clear_sentence)
clear_button.grid(row=0, column=2, padx=10)

space_button = Button(button_frame, text="Add Space", width=15, font=("Helvetica", 12), bg="#ffa500", fg="white", command=add_space)
space_button.grid(row=1, column=0, padx=10, pady=5)

new_word_button = Button(button_frame, text="New Word", width=15, font=("Helvetica", 12), bg="#8a2be2", fg="white", command=start_new_word)
new_word_button.grid(row=1, column=1, padx=10, pady=5)

undo_button = Button(button_frame, text="Undo Last Word", width=15, font=("Helvetica", 12), bg="#ff6347", fg="white", command=undo_last_word)
undo_button.grid(row=1, column=2, padx=10, pady=5)

# Styling for progress bar
style = Style()
style.configure("green.Horizontal.TProgressbar", troughcolor="#f5f5f5", background="#4caf50", thickness=20)

# OpenCV and Hand Recognition in a separate thread
def run_opencv():
    global predicted_letter
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])

                    landmarks = np.array(landmarks).reshape(1, -1)
                    prediction = clf.predict(landmarks)[0]
                    confidence = clf.predict_proba(landmarks).max() * 100  # Calculate confidence

                    predicted_letter = prediction
                    detected_letter.set(f"Detected Letter: {predicted_letter}")
                    confidence_level.set(f"Confidence: {confidence:.2f}%")
                    progress["value"] = confidence

            cv2.imshow("Sign Language Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

# Run OpenCV in a separate thread
threading.Thread(target=run_opencv, daemon=True).start()

# Start the Tkinter main loop
root.mainloop()