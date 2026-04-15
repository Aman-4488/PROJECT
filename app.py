import streamlit as st
import cv2
import dlib
import numpy as np
import os

# ---------------- CONFIG ----------------
MODEL_DIR = "models"
KNOWN_FACES_DIR = "known_faces"
THRESHOLD = 0.6

os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

# ---------------- LOAD MODELS ----------------
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(
    os.path.join(MODEL_DIR, "shape_predictor_68_face_landmarks.dat")
)
facerec = dlib.face_recognition_model_v1(
    os.path.join(MODEL_DIR, "dlib_face_recognition_resnet_model_v1.dat")
)

# ---------------- LOAD KNOWN FACES ----------------
known_embeddings = []
known_names = []

for name in os.listdir(KNOWN_FACES_DIR):
    person_dir = os.path.join(KNOWN_FACES_DIR, name)
    if os.path.isdir(person_dir):
        for file in os.listdir(person_dir):
            emb = np.load(os.path.join(person_dir, file))
            known_embeddings.append(emb)
            known_names.append(name)


def detect_emotion(shape):
    points = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)], dtype=np.float32)

    face_width = np.linalg.norm(points[16] - points[0]) + 1e-6
    face_height = np.linalg.norm(points[8] - points[27]) + 1e-6
    mouth_width = np.linalg.norm(points[54] - points[48]) + 1e-6
    mouth_open = np.linalg.norm(points[66] - points[62])
    smile_ratio = mouth_width / face_width
    mouth_open_ratio = mouth_open / mouth_width
    corner_drop = (((points[48][1] + points[54][1]) / 2) - points[51][1]) / face_height

    if mouth_open_ratio > 0.38:
        return "Surprised"
    if smile_ratio > 0.42 and corner_drop < 0.015:
        return "Happy"
    if corner_drop > 0.03 and smile_ratio < 0.40:
        return "Sad"
    return "Neutral"

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Face Recognition", layout="wide")
st.title("🔴 Real-Time Face Recognition (Dlib)")
st.write("Unknown face → enter name → saved automatically")

name_input = st.text_input("Enter name for unknown face (press Enter)")

run = st.checkbox("Start Camera")

FRAME_WINDOW = st.image([])

# ---------------- CAMERA LOOP ----------------
if run:
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera not accessible")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector(rgb)

        for face in faces:
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()

            shape = sp(rgb, face)
            embedding = np.array(facerec.compute_face_descriptor(rgb, shape))
            emotion = detect_emotion(shape)

            name = "Unknown"

            if known_embeddings:
                known_embeddings_array = np.array(known_embeddings)
                dists = np.linalg.norm(known_embeddings_array - embedding, axis=1)
                idx = np.argmin(dists)

                if dists[idx] < THRESHOLD:
                    name = known_names[idx]

            # -------- SAVE UNKNOWN FACE --------
            if name == "Unknown" and name_input.strip() != "":
                person_path = os.path.join(KNOWN_FACES_DIR, name_input)
                os.makedirs(person_path, exist_ok=True)

                file_id = len(os.listdir(person_path))
                np.save(os.path.join(person_path, f"{file_id}.npy"), embedding)

                known_embeddings.append(embedding)
                known_names.append(name_input)

                name = name_input
                st.success(f"Saved face for {name_input}")

            # -------- DRAW RECTANGLE --------
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"{name} | {emotion}"
            cv2.putText(
                frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2
            )

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
