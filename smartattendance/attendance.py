import cv2
import numpy as np
import os
import csv
from datetime import datetime

# Load Haar Cascade (comes with OpenCV)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Paths
DATASET_PATH = "dataset"
ATTENDANCE_FILE = "attendance.csv"

# Ensure dataset folder exists
if not os.path.exists(DATASET_PATH):
    os.makedirs(DATASET_PATH)

def register_student(name):
    """Register a new student by capturing face images."""
    cap = cv2.VideoCapture(0)
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            face = gray[y:y + h, x:x + w]
            cv2.imwrite(f"{DATASET_PATH}/{name}_{count}.jpg", face)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Register Student", frame)

        if cv2.waitKey(1) & 0xFF == ord("q") or count >= 20:  # Capture 20 images
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Registration complete for {name}")

def mse(imageA, imageB):
    """Calculate Mean Squared Error between two images."""
    if imageA.shape != imageB.shape:
        return float("inf")
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def recognize_and_mark():
    """Recognize student by comparing faces and mark attendance."""
    known_faces = {}
    
    # Load dataset
    for file in os.listdir(DATASET_PATH):
        if file.endswith(".jpg"):
            name = file.split("_")[0]
            img = cv2.imread(os.path.join(DATASET_PATH, file), cv2.IMREAD_GRAYSCALE)
            if name not in known_faces:
                known_faces[name] = []
            known_faces[name].append(img)

    cap = cv2.VideoCapture(0)
    marked = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            face_resized = cv2.resize(face, (100, 100))

            best_match = None
            min_error = float("inf")

            # Compare with known faces
            for name, imgs in known_faces.items():
                for ref_img in imgs:
                    ref_resized = cv2.resize(ref_img, (100, 100))
                    error = mse(face_resized, ref_resized)
                    if error < min_error:
                        min_error = error
                        best_match = name

            # If matched
            if best_match and min_error < 2000:  # Threshold
                cv2.putText(frame, best_match, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                if best_match not in marked:
                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    with open(ATTENDANCE_FILE, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([best_match, now])
                    marked.add(best_match)
                    print(f"[INFO] Attendance marked for {best_match}")

        cv2.imshow("Attendance System", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# ---------------- MAIN ----------------
print("Options:")
print("1. Register Student")
print("2. Start Attendance")
choice = input("Enter choice (1/2): ")

if choice == "1":
    student_name = input("Enter student name: ")
    register_student(student_name)
elif choice == "2":
    recognize_and_mark()
else:
    print("Invalid choice")
