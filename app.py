import streamlit as st
import cv2
import dlib
import numpy as np
import os

# ---------------- CONFIG ----------------
MODEL_DIR = "models"
KNOWN_FACES_DIR = "known_faces"
THRESHOLD = 0.6
EPSILON_AVOID_ZERO_DIVISION = 1e-6
# Emotion heuristic thresholds tuned for webcam framing with dlib landmarks.
# They are normalized against face geometry to keep behavior stable across scales.
EMOTION_OPEN_MOUTH_THRESHOLD = 0.38
EMOTION_SMILE_THRESHOLD = 0.42
EMOTION_HAPPY_CORNER_DROP_MAX = 0.015
EMOTION_SAD_CORNER_DROP_MIN = 0.03
EMOTION_SAD_SMILE_MAX = 0.40
LANDMARK_JAW_LEFT = 0
LANDMARK_CHIN = 8
LANDMARK_JAW_RIGHT = 16
LANDMARK_NOSE_BRIDGE_TOP = 27
LANDMARK_MOUTH_LEFT = 48
LANDMARK_UPPER_LIP_CENTER = 51
LANDMARK_MOUTH_RIGHT = 54
LANDMARK_INNER_UPPER_LIP = 62
LANDMARK_INNER_LOWER_LIP = 66
REQUIRED_EMOTION_LANDMARKS = (
    LANDMARK_JAW_LEFT,
    LANDMARK_CHIN,
    LANDMARK_JAW_RIGHT,
    LANDMARK_NOSE_BRIDGE_TOP,
    LANDMARK_MOUTH_LEFT,
    LANDMARK_UPPER_LIP_CENTER,
    LANDMARK_MOUTH_RIGHT,
    LANDMARK_INNER_UPPER_LIP,
    LANDMARK_INNER_LOWER_LIP,
)

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

known_embeddings = (
    np.array(known_embeddings, dtype=np.float32)
    if known_embeddings
    else np.empty((0, 128), dtype=np.float32)
)


def detect_emotion(shape):
    points = {}
    for idx in REQUIRED_EMOTION_LANDMARKS:
        part = shape.part(idx)
        points[idx] = np.array([part.x, part.y], dtype=np.float32)

    face_width = np.linalg.norm(points[LANDMARK_JAW_RIGHT] - points[LANDMARK_JAW_LEFT])
    face_height = np.linalg.norm(points[LANDMARK_CHIN] - points[LANDMARK_NOSE_BRIDGE_TOP])
    mouth_width = np.linalg.norm(points[LANDMARK_MOUTH_RIGHT] - points[LANDMARK_MOUTH_LEFT])
    mouth_open = np.linalg.norm(points[LANDMARK_INNER_LOWER_LIP] - points[LANDMARK_INNER_UPPER_LIP])
    smile_ratio = mouth_width / max(face_width, EPSILON_AVOID_ZERO_DIVISION)
    mouth_open_ratio = mouth_open / max(mouth_width, EPSILON_AVOID_ZERO_DIVISION)
    # Measures how far mouth corners droop below the upper-lip center relative to face height.
    corner_drop = (
        (
            (points[LANDMARK_MOUTH_LEFT][1] + points[LANDMARK_MOUTH_RIGHT][1]) / 2
        ) - points[LANDMARK_UPPER_LIP_CENTER][1]
    ) / max(face_height, EPSILON_AVOID_ZERO_DIVISION)

    if mouth_open_ratio > EMOTION_OPEN_MOUTH_THRESHOLD:
        return "Surprised"
    if smile_ratio > EMOTION_SMILE_THRESHOLD and corner_drop < EMOTION_HAPPY_CORNER_DROP_MAX:
        return "Happy"
    if corner_drop > EMOTION_SAD_CORNER_DROP_MIN and smile_ratio < EMOTION_SAD_SMILE_MAX:
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

            if known_embeddings.size:
                dists = np.linalg.norm(known_embeddings - embedding, axis=1)
                idx = np.argmin(dists)

                if dists[idx] < THRESHOLD:
                    name = known_names[idx]

            # -------- SAVE UNKNOWN FACE --------
            if name == "Unknown" and name_input.strip() != "":
                person_path = os.path.join(KNOWN_FACES_DIR, name_input)
                os.makedirs(person_path, exist_ok=True)

                file_id = len(os.listdir(person_path))
                np.save(os.path.join(person_path, f"{file_id}.npy"), embedding)

                known_embeddings = np.vstack([known_embeddings, embedding])
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
